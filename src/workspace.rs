use std::{
    cell::{Cell, OnceCell},
    collections::BTreeMap,
};

use anyhow::{Context, Result};

use crate::{analyze, syntax::SyntaxNodeMatch as _};

#[derive(Default)]
pub struct Workspace {
    pub documents: BTreeMap<lsp::Uri, Document>,
}

impl Workspace {
    pub(crate) fn document(&self, uri: &lsp::Uri) -> Result<&Document> {
        self.documents
            .get(uri)
            .with_context(|| format!("no such open document: {:?}", uri.as_str()))
    }

    pub(crate) fn document_mut(&mut self, uri: &lsp::Uri) -> Result<&mut Document> {
        self.documents
            .get_mut(uri)
            .with_context(|| format!("no such open document: {:?}", uri.as_str()))
    }

    pub(crate) fn create_document(&mut self, new: lsp::TextDocumentItem) -> &mut Document {
        let document = self.documents.entry(new.uri).or_default();
        document.content = new.text;
        document.version = Some(new.version);
        document.language = DocumentLanguage::from_str(&new.language_id);
        document.reset();
        document
    }

    pub(crate) fn remove_document(&mut self, text_document: lsp::TextDocumentIdentifier) {
        self.documents.remove(&text_document.uri);
    }
}

#[derive(Default)]
pub struct Document {
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
    #[cfg(test)]
    pub fn new(content: String) -> Document {
        Document { content, ..Default::default() }
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

    pub fn global_scope(&self) -> &GlobalScopeOutput {
        self.global_scope.get_or_init(|| {
            let parsed = self.parse();
            let (symbols, errors) =
                crate::analyze::collect_global_scope(&parsed.tree, self.content());
            GlobalScopeOutput { symbols, errors }
        })
    }

    pub fn symbol_at_offset(
        &self,
        offset: OffsetUtf8,
    ) -> Option<(crate::analyze::ResolvedSymbol, crate::parse::NodeIndex)> {
        let parsed = self.parse();

        let token_path = parsed.tree.token_path_at_offset_utf8(offset);
        let token = *token_path.last()?;
        assert_eq!(token_path[0], parsed.tree.root_index());

        let decl_index = *token_path.get(1)?;
        let decl = crate::syntax::Decl::from_tree(&parsed.tree, decl_index)?;
        if let Some(name) = decl.name(&parsed.tree) {
            if name.index() == token {
                let text = &self.content[name.byte_range()];
                let symbol = crate::analyze::ResolvedSymbol {
                    name: text.into(),
                    reference: crate::analyze::Reference::User(decl.into()),
                };
                return Some((symbol, token));
            }
        }

        let global_scope = &self.global_scope().symbols;
        let mut context = analyze::Context::new(global_scope, &parsed.tree, &self.content);
        context.analyze_decl(decl);

        Some((context.get_node_symbol(token)?, token))
    }

    pub fn visible_symbols_at_offset(
        &self,
        offset: OffsetUtf8,
    ) -> Option<(Vec<crate::analyze::ResolvedSymbol>, crate::parse::NodeIndex)> {
        let parsed = self.parse();

        let token_path = parsed.tree.token_path_at_offset_utf8(offset);
        let token = *token_path.last()?;
        assert_eq!(token_path[0], parsed.tree.root_index());

        let decl_index = *token_path.get(1)?;
        let decl = crate::syntax::Decl::from_tree(&parsed.tree, decl_index)?;

        let global_scope = &self.global_scope().symbols;
        let mut context = analyze::Context::new(global_scope, &parsed.tree, &self.content);

        context.capture_symbols_at(token);
        context.analyze_decl(decl);
        let symbols = context.get_captured_symbols();

        Some((symbols, token))
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
        let document = Document::new("abc\n123\nfn main() {}\n".into());
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
