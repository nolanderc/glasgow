use crate::parse::{self, token::TokenSet, Tag};

pub fn format(tree: &parse::Tree, source: &str) -> String {
    let mut formatter = Formatter::new(tree, source);
    formatter.emit_node(tree.root_index());

    while formatter.output.ends_with(char::is_whitespace) {
        formatter.output.pop();
    }

    if !formatter.output.is_empty() {
        formatter.emit_newlines(1, 1);
    }

    formatter.output
}

struct Formatter<'a> {
    source: &'a str,
    tree: &'a parse::Tree,
    output: String,
    previous_token: Tag,
    next_extra: usize,
    last_emit: usize,

    indent: usize,
    needs_indent: bool,
}

impl<'a> Formatter<'a> {
    fn new(tree: &'a parse::Tree, source: &'a str) -> Self {
        Self {
            source,
            tree,
            output: String::new(),
            previous_token: Tag::Eof,
            next_extra: 0,
            last_emit: 0,
            indent: 0,
            needs_indent: false,
        }
    }

    fn indent_decrease(&mut self) {
        self.indent = self.indent.saturating_sub(4);
    }

    fn indent_increase(&mut self) {
        self.indent = self.indent.saturating_add(4);
    }

    fn emit_node(&mut self, index: parse::NodeIndex) {
        let node = self.tree.node(index);

        let indent_before = self.indent;

        match node.tag() {
            Tag::Root => {
                for child in node.children() {
                    self.emit_node(child);
                    self.emit_newlines(1, 2);
                }
            },

            Tag::StmtBlock => {
                let count = node.children().count();

                let multiline = if count > 3 {
                    true
                } else if let Some(range) = self.tree.byte_range_total(index) {
                    self.source[range].find('\n').is_some()
                } else {
                    true
                };

                for (i, child) in node.children().enumerate() {
                    if i == count - 1 && self.tree.node(child).tag() == Tag::RCurly {
                        self.indent_decrease();
                        if !multiline {
                            self.emit_space();
                        }
                    }

                    self.emit_node(child);

                    if i == 0 {
                        self.indent_increase();
                        if !multiline {
                            self.emit_space();
                        }
                    }

                    if multiline && i < count - 1 {
                        self.emit_newlines(1, 2);
                    }
                }
            },

            Tag::ArgumentList | Tag::DeclFnParameterList => {
                self.emit_list_comma_separated(node, Tag::LParen, Tag::RParen);
            },
            Tag::TemplateList => {
                self.emit_list_comma_separated(node, Tag::TemplateListStart, Tag::TemplateListEnd);
            },

            Tag::DeclStructFieldList => {
                let count = node.children().count();
                for (i, child) in node.children().enumerate() {
                    if i == count - 1 && self.tree.node(child).tag() == Tag::RCurly {
                        self.indent_decrease();
                    }

                    self.emit_node(child);

                    if i == 0 {
                        self.indent_increase();
                        self.emit_newlines(1, 1);
                    } else if i < count - 1 {
                        self.emit_newlines(1, 2);
                    }
                }
            },

            Tag::DeclStructField => {
                let mut has_comma = false;
                for child in node.children() {
                    self.emit_node(child);
                    has_comma |= self.tree.node(child).tag() == Tag::Comma;
                }
                if !has_comma {
                    self.emit_text(",");
                }
            },

            tag if tag.is_token() => self.emit_token(node),
            _ => {
                for child in node.children() {
                    self.emit_node(child);
                }
            },
        }

        if matches!(node.tag(), Tag::AttributeList) {
            self.emit_space_or_newlines(1);
        }

        self.indent = indent_before;
    }

    fn emit_list_comma_separated(&mut self, node: parse::Node, open_tag: Tag, close_tag: Tag) {
        let count = node.children().count();

        let open =
            node.children().next().map(|x| self.tree.node(x)).filter(|x| x.tag() == open_tag);
        let close =
            node.children().next_back().map(|x| self.tree.node(x)).filter(|x| x.tag() == close_tag);

        let mut items = node.children();
        if open.is_some() {
            items.next();
        }
        if close.is_some() {
            items.next_back();
        }

        let multiline = if let Some(range) = self.tree.byte_range_total_children(items.clone()) {
            self.source[range].ends_with(',')
        } else {
            false
        };

        for (i, child) in node.children().enumerate() {
            if i == count - 1 && self.tree.node(child).tag() == close_tag {
                self.indent_decrease();
            }

            self.emit_node(child);

            if i == 0 {
                self.indent_increase();
            }

            if multiline && i < count - 1 {
                self.emit_newlines(1, 1);
            }
        }
    }

    fn emit_token(&mut self, node: parse::Node) {
        let always_space_before = const {
            parse::EXPRESSION_INFIX_OPS.union(parse::ASSIGNMENT_OPS).with_many(&[
                Tag::Identifier,
                Tag::ThinArrowRight,
                Tag::LCurly,
                Tag::AtSign,
            ])
        };

        let always_space_after = const {
            parse::EXPRESSION_INFIX_OPS
                .union(parse::ASSIGNMENT_OPS)
                .union(Tag::NUMBERS)
                .union(Tag::KEYWORDS)
                .with_many(&[Tag::Identifier, Tag::ThinArrowRight, Tag::Comma, Tag::Colon])
        };

        let never_space_before = const {
            TokenSet::new(&[
                Tag::LParen,
                Tag::RParen,
                Tag::TemplateListStart,
                Tag::TemplateListEnd,
                Tag::Dot,
                Tag::Comma,
                Tag::Colon,
                Tag::SemiColon,
            ])
        };

        let never_space_after =
            const { TokenSet::new(&[Tag::LParen, Tag::TemplateListStart, Tag::Dot, Tag::AtSign]) };

        let never = || {
            never_space_before.contains(node.tag())
                || never_space_after.contains(self.previous_token)
        };

        let always = || {
            always_space_before.contains(node.tag())
                || always_space_after.contains(self.previous_token)
        };

        let exception = || match (self.previous_token, node.tag()) {
            (Tag::KeywordVar, Tag::TemplateListStart) => false,
            (keyword, _) if keyword.is_keyword() => true,
            _ => false,
        };

        if (!never() && always()) || exception() {
            self.emit_space();
        }

        self.emit_range(node.byte_range(), 0);

        self.previous_token = node.tag();
    }

    fn emit_comments_before(&mut self, offset: usize, max_lines_after: usize) -> Option<usize> {
        let mut lines = None;

        while let Some(extra) = self.tree.extra(self.next_extra) {
            let extra_range = extra.byte_range();
            if extra_range.start < offset {
                self.next_extra += 1;
                self.emit_space_or_newlines(2);

                let line_start =
                    self.source[..extra_range.start].rfind('\n').map(|x| x + 1).unwrap_or(0);
                let leading_whitespace = extra_range.start - line_start;
                self.emit_range(extra_range, leading_whitespace);
                lines = Some(self.emit_space_or_newlines(max_lines_after));
            } else {
                break;
            }
        }

        lines
    }

    fn emit_range(&mut self, range: std::ops::Range<usize>, ignore_leading_whitespace: usize) {
        self.emit_comments_before(range.start, 2);

        self.last_emit = range.end;
        let mut text = &self.source[range];

        let trim_leading_whitespace = |line: &'a str| {
            let leading_whitespace = line.len() - line.trim_start().len();
            &line[usize::min(leading_whitespace, ignore_leading_whitespace)..]
        };

        while let Some(newline) = text.find('\n') {
            let (line, rest) = text.split_at(newline);
            self.emit_text(trim_leading_whitespace(line));
            self.emit_newline_raw();
            text = &rest[1..];
        }

        self.emit_text(trim_leading_whitespace(text));
    }

    fn emit_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }

        if self.needs_indent {
            for _ in 0..self.indent {
                self.output.push(' ');
            }
            self.needs_indent = false;
        }

        self.output += text;
    }

    fn emit_space_or_newlines(&mut self, max: usize) -> usize {
        let lines = self.emit_newlines(0, max);
        if lines == 0 {
            self.emit_space();
        }
        lines
    }

    fn emit_newlines(&mut self, min: usize, max: usize) -> usize {
        let remainder = &self.source[self.last_emit..];
        let trimmed = remainder.trim_start();
        let whitespace = &remainder[..remainder.len() - trimmed.len()];
        let mut count = whitespace.chars().filter(|x| *x == '\n').count();

        self.last_emit += whitespace.len();

        let mut emitted = 0;
        if count == 0 && min != 0 {
            if let Some(lines_after) = self.emit_comments_before(self.last_emit + 1, max) {
                emitted = lines_after;
            }
        }

        count = count.clamp(min, max);

        for _ in emitted..count {
            self.emit_newline_raw();
        }

        count
    }

    fn emit_space(&mut self) {
        if self.output.chars().next_back().map(|x| x.is_whitespace()).unwrap_or(true) {
            return;
        }
        self.output += " ";
    }

    fn emit_newline_raw(&mut self) {
        // ensure lines don't end with whitespace
        while self.output.ends_with(' ') {
            self.output.pop();
        }
        self.output += "\n";
        self.needs_indent = true;
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use indoc::indoc;

    use super::*;

    fn check_formatting(source: &str, expected: expect_test::Expect) {
        let mut parser = parse::Parser::new();
        let tree = &parse::parse_file(&mut parser, source).tree;
        let formatted = format(tree, source);
        expected.assert_eq(&formatted);
    }

    #[test]
    fn large() {
        check_formatting(
            indoc! {r#"
                // this file contains a bunch of various syntax constructs

                @fragment
                fn fs1() -> @location(0) vec4<f32> {
                    return vec4(1.0, 0.0, 0.0, 1.0);
                }

                // line-wrap parameters that end with a ','
                fn foo(x: u32, y: u32,) -> vec4<f32> {
                    // line-wrap arguments that end with a ','
                    return vec4(1.0, 0.0, 0.0, 1.0,);

                    // "unwrap" arguments that don't end with a ','
                    return vec4(
                        1.0,
                        0.0,
                        0.0,
                        1.0
                    );

                    /* this is just a comment */

                        /* this comment is
                         * split across multiple
                         * lines, but the stars should
                         * still be aligned :-) */

                    let y = vec2<f32>(123, 456);

                    // single-line blocks should be allowed
                    if (1 + 1 == 2) { return vec4(); } 
                }

                struct Uniforms {
                    x: f32,
                    y: f32 // missing commas should be inserted

                    // multiline type specifiers are allowed if you really want
                    complex_array: array<vec2<f32>, 1234,>
                }

                @group(0) @binding(0)
                var<uniform> uniforms: Uniforms;
            "#},
            expect![[r#"
                // this file contains a bunch of various syntax constructs

                @fragment
                fn fs1() -> @location(0) vec4<f32> {
                    return vec4(1.0, 0.0, 0.0, 1.0);
                }

                // line-wrap parameters that end with a ','
                fn foo(
                    x: u32,
                    y: u32,
                ) -> vec4<f32> {
                    // line-wrap arguments that end with a ','
                    return vec4(
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                    );

                    // "unwrap" arguments that don't end with a ','
                    return vec4(1.0, 0.0, 0.0, 1.0);

                    /* this is just a comment */

                    /* this comment is
                     * split across multiple
                     * lines, but the stars should
                     * still be aligned :-) */

                    let y = vec2<f32>(123, 456);

                    // single-line blocks should be allowed
                    if (1 + 1 == 2) { return vec4(); }
                }

                struct Uniforms {
                    x: f32,
                    y: f32, // missing commas should be inserted

                    // multiline type specifiers are allowed if you really want
                    complex_array: array<
                        vec2<f32>,
                        1234,
                    >,
                }

                @group(0) @binding(0)
                var<uniform> uniforms: Uniforms;
            "#]],
        )
    }
}
