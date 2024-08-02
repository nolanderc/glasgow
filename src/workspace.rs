use std::{
    cell::{Cell, OnceCell},
    collections::BTreeMap,
};

use anyhow::{Context, Result};

use crate::{
    analyze,
    arena::{Arena, Handle},
    syntax::SyntaxNodeMatch as _,
};

#[derive(Default)]
pub struct Workspace {
    document_ids: BTreeMap<lsp::Uri, DocumentId>,
    documents: Arena<Document>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DocumentId(Handle<Document>);

impl Workspace {
    pub(crate) fn create_document(&mut self, new: lsp::TextDocumentItem) -> &mut Document {
        let id = *self.document_ids.entry(new.uri).or_insert_with(|| {
            DocumentId(
                self.documents.insert_with_handle(|handle| Document::new(DocumentId(handle))),
            )
        });

        let document = &mut self.documents[id.0];
        document.content = new.text;
        document.version = Some(new.version);
        document.language = DocumentLanguage::from_str(&new.language_id);
        document.reset();
        document
    }

    pub(crate) fn remove_document(&mut self, text_document: lsp::TextDocumentIdentifier) {
        let Some(id) = self.document_ids.remove(&text_document.uri) else { return };
        self.documents.remove(id.0);
    }

    pub(crate) fn document_mut(&mut self, uri: &lsp::Uri) -> Result<&mut Document> {
        let id = *self
            .document_ids
            .get_mut(uri)
            .with_context(|| format!("no such open document: {:?}", uri.as_str()))?;
        Ok(&mut self.documents[id.0])
    }

    pub(crate) fn document(&self, uri: &lsp::Uri) -> Result<&Document> {
        let id = *self
            .document_ids
            .get(uri)
            .with_context(|| format!("no such open document: {:?}", uri.as_str()))?;
        Ok(&self.documents[id.0])
    }

    pub(crate) fn document_from_id(&self, id: DocumentId) -> &Document {
        &self.documents[id.0]
    }

    pub fn format_type<'a>(&'a self, typ: &'a crate::analyze::Type) -> impl std::fmt::Display + 'a {
        crate::util::fmt_from_fn(move |f| typ.fmt(f, self))
    }
}

pub struct Document {
    id: DocumentId,

    content: String,
    version: Option<DocumentVersion>,
    language: Option<DocumentLanguage>,

    parser: Cell<crate::parse::Parser<'static>>,
    parser_output: OnceCell<crate::parse::Output>,

    global_scope: OnceCell<GlobalScopeOutput>,
}

pub struct GlobalScopeOutput {
    pub symbols: crate::analyze::GlobalScope,
    pub errors: BTreeMap<String, crate::analyze::ErrorDuplicate>,
}

impl Document {
    pub fn new(id: DocumentId) -> Document {
        Document {
            id,
            content: Default::default(),
            version: None,
            language: None,
            parser: Default::default(),
            parser_output: Default::default(),
            global_scope: Default::default(),
        }
    }

    /// Called when the document contents change.
    /// This clears any cached values.
    pub fn reset(&mut self) {
        self.global_scope.take();
        if let Some(output) = self.parser_output.take() {
            // reuse storage from previous tree
            self.parser.set(self.parser.take().reset(output));
        }
    }

    pub(crate) fn apply_change(
        &mut self,
        change: lsp::TextDocumentContentChangeEvent,
    ) -> Result<()> {
        match change.range {
            None => {
                self.content = change.text;
                Ok(())
            },

            Some(range) => {
                let start =
                    self.offset_utf8_from_position_utf16(range.start).context("invalid range")?;
                let end =
                    self.offset_utf8_from_position_utf16(range.end).context("invalid range")?;
                self.content.replace_range(start..end, &change.text);
                Ok(())
            },
        }
    }

    pub fn id(&self) -> DocumentId {
        self.id
    }

    pub fn version(&self) -> Option<i32> {
        self.version
    }

    pub(crate) fn set_version(&mut self, version: i32) {
        self.version = Some(version);
    }

    pub(crate) fn content(&self) -> &str {
        &self.content
    }

    pub fn offset_utf8_from_position_utf16(&self, position: lsp::Position) -> Option<OffsetUtf8> {
        let mut lines = std::iter::from_fn({
            let mut text = self.content.as_str();
            move || {
                let mut len = 0;
                let bytes = text.as_bytes();
                while len < bytes.len() {
                    let byte = bytes[len];
                    len += 1;
                    if byte == b'\n' {
                        break;
                    }
                }
                let (line, rest) = text.split_at(len);
                text = rest;
                if len == 0 {
                    None
                } else {
                    Some(line)
                }
            }
        });

        let mut offset_line = 0;
        for _ in 0..position.line {
            offset_line += lines.next()?.len();
        }
        let line = &self.content[offset_line..];

        let mut offset_character = 0;
        let mut characters_remaining = position.character as usize;
        for (start, codepoint) in line.char_indices() {
            if codepoint == '\n' {
                return None;
            }

            let len = codepoint.len_utf16();
            if len > characters_remaining {
                characters_remaining = 0;
                break;
            }
            characters_remaining -= len;
            offset_character = start + codepoint.len_utf8();

            if characters_remaining == 0 {
                break;
            }
        }

        if characters_remaining == 0 {
            Some(offset_line + offset_character)
        } else {
            None
        }
    }

    pub fn position_utf16_from_offset_utf8(&self, offset: OffsetUtf8) -> Option<lsp::Position> {
        let before = self.content.get(..offset)?;
        let line_count = before.bytes().filter(|x| *x == b'\n').count();
        let line_start = before.rfind('\n').map(|x| x + 1).unwrap_or(0);
        let line = &before[line_start..];
        let character_offset = line.chars().map(|x| x.len_utf16()).sum::<usize>();
        Some(lsp::Position { line: line_count as u32, character: character_offset as u32 })
    }

    pub(crate) fn range_utf16_from_range_utf8(
        &self,
        range: std::ops::Range<OffsetUtf8>,
    ) -> Option<lsp::Range> {
        Some(lsp::Range {
            start: self.position_utf16_from_offset_utf8(range.start)?,
            end: self.position_utf16_from_offset_utf8(range.end)?,
        })
    }

    pub fn parse(&self) -> &crate::parse::Output {
        self.parser_output.get_or_init(|| {
            let mut parser = self.parser.take();
            let output = std::mem::take(crate::parse::parse_file(&mut parser, &self.content));
            self.parser.set(parser.reset(crate::parse::Output::default()));
            output
        })
    }

    pub fn global_scope(&self, workspace: &Workspace) -> &GlobalScopeOutput {
        self.global_scope.get_or_init(|| {
            let (symbols, errors) = crate::analyze::collect_global_scope(workspace, self.id);
            GlobalScopeOutput { symbols, errors }
        })
    }

    pub fn symbol_in_range(
        &self,
        workspace: &Workspace,
        range: std::ops::Range<usize>,
    ) -> Option<(crate::analyze::ResolvedSymbol, crate::parse::NodeIndex)> {
        let parsed = self.parse();

        let token_path = parsed.tree.token_path_in_range_utf8(range);
        let token = *token_path.last()?;
        assert_eq!(token_path[0], parsed.tree.root_index());

        let decl_index = *token_path.get(1)?;
        let decl = crate::syntax::Decl::from_tree(&parsed.tree, decl_index)?;

        let mut context = analyze::DocumentContext::new(workspace, self);
        context.analyze_decl(decl);

        Some((context.get_node_symbol(token)?, token))
    }

    pub fn visible_symbols_in_range(
        &self,
        workspace: &Workspace,
        range: std::ops::Range<usize>,
    ) -> Option<(Vec<crate::analyze::ResolvedSymbol>, crate::parse::NodeIndex)> {
        let parsed = self.parse();

        let token_path = parsed.tree.token_path_in_range_utf8(range);
        let token = *token_path.last()?;
        assert_eq!(token_path[0], parsed.tree.root_index());

        let decl_index = *token_path.get(1)?;
        let decl = crate::syntax::Decl::from_tree(&parsed.tree, decl_index)?;

        let mut context = analyze::DocumentContext::new(workspace, self);

        let mut options = analyze::CaptureOptions::default();
        for &index in token_path.iter() {
            let node = parsed.tree.node(index);
            match node.tag() {
                crate::parse::Tag::Attribute => options.attributes = true,
                crate::parse::Tag::ArgumentList => options.attributes = false,
                crate::parse::Tag::TemplateList => options.template_arguments = true,
                _ => continue,
            }
        }
        context.capture_symbols_at(token, options);

        context.analyze_decl(decl);
        let symbols = context.get_captured_symbols();

        Some((symbols, token))
    }

    pub(crate) fn find_all_references(
        &self,
        workspace: &Workspace,
        reference: &analyze::Reference,
    ) -> Vec<crate::parse::NodeIndex> {
        let parsed = self.parse();
        let mut context = analyze::DocumentContext::new(workspace, self);
        for decl in crate::syntax::root(&parsed.tree).decls(&parsed.tree) {
            context.analyze_decl(decl);
        }
        context.find_all_references(reference)
    }
}

pub type OffsetUtf8 = usize;

pub type DocumentVersion = i32;

pub enum DocumentLanguage {
    Wgsl,
}

impl DocumentLanguage {
    pub fn from_str(x: &str) -> Option<DocumentLanguage> {
        if x.eq_ignore_ascii_case("wgsl") {
            Some(DocumentLanguage::Wgsl)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_offset() {
        let mut workspace = Workspace::default();
        let document = workspace.create_document(lsp::TextDocumentItem {
            uri: "file://foo".parse().unwrap(),
            language_id: "wgsl".into(),
            version: 0,
            text: "abc\n123\nfn main() {}\n".into(),
        });

        let check_utf8_from_utf16 = |(line, character), expected: Option<usize>| {
            let position = lsp::Position::new(line, character);
            let offset = document.offset_utf8_from_position_utf16(position);
            assert_eq!(offset, expected, "at {position:?}");
        };

        check_utf8_from_utf16((0, 0), Some(0));
        check_utf8_from_utf16((0, 1), Some(1));
        check_utf8_from_utf16((0, 2), Some(2));
        check_utf8_from_utf16((0, 3), Some(3));

        check_utf8_from_utf16((1, 0), Some(4));
        check_utf8_from_utf16((1, 1), Some(5));
        check_utf8_from_utf16((1, 2), Some(6));
        check_utf8_from_utf16((1, 3), Some(7));

        check_utf8_from_utf16((2, 0), Some(8));
        check_utf8_from_utf16((2, 1), Some(9));
        check_utf8_from_utf16((2, 2), Some(10));
        check_utf8_from_utf16((2, 3), Some(11));

        check_utf8_from_utf16((3, 0), Some(21));
        check_utf8_from_utf16((3, 1), None);

        check_utf8_from_utf16((4, 0), None);
    }
}
