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

    state: State,
}

#[derive(Clone)]
struct State {
    previous_token: Tag,
    next_extra: usize,
    last_emit: usize,

    line: usize,

    /// If set to `true`, the next token emitted will not have any leading space inserted.
    force_no_space_before: bool,

    indent: usize,
    indent_base: usize,
    needs_indent: bool,
}

impl<'a> Formatter<'a> {
    fn new(tree: &'a parse::Tree, source: &'a str) -> Self {
        Self {
            source,
            tree,
            output: String::new(),
            state: State {
                previous_token: Tag::Eof,
                next_extra: 0,
                last_emit: 0,
                line: 0,
                force_no_space_before: false,
                indent: 0,
                indent_base: 0,
                needs_indent: false,
            },
        }
    }

    fn indent_decrease(&mut self) {
        self.state.indent = self.state.indent.saturating_sub(4);
    }

    fn indent_increase(&mut self) {
        self.state.indent = self.state.indent.saturating_add(4);
    }

    fn emit_node(&mut self, index: parse::NodeIndex) {
        let node = self.tree.node(index);

        let indent_before = self.state.indent;

        match node.tag() {
            Tag::Root => {
                for child in node.children() {
                    self.emit_node(child);
                    self.emit_newlines(1, 3);
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

                    let base_old =
                        std::mem::replace(&mut self.state.indent_base, self.state.indent);
                    self.emit_node(child);
                    self.state.indent_base = base_old;

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

            Tag::ExprPrefix => {
                for (i, child) in node.children().enumerate() {
                    if i != 0 {
                        self.state.force_no_space_before = true;
                    }
                    self.emit_node(child);
                }
                self.state.force_no_space_before = false;
            },

            Tag::ExprInfix => {
                for child in node.children() {
                    let node = self.tree.node(child);
                    if node.is_token()
                        && parse::EXPRESSION_INFIX_OPS.contains(node.tag())
                        && self.emit_newlines(0, 1) != 0
                    {
                        self.state.indent = self.state.indent_base + 4;
                    }
                    self.emit_node(child);
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

        self.state.indent = indent_before;
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

        let mut multiline = false;
        let mut multicolumn = false;

        if let Some(range) = self.tree.byte_range_total_children(items.clone()) {
            multiline = self.source[range.clone()].ends_with(',');
            multicolumn = self.source[range].contains('\n');
        };

        let state = self.state.clone();
        let output_len = self.output.len();
        let mut column_sizes = Vec::with_capacity(8);
        let mut retries = 1;

        'outer: loop {
            let mut column = 0;
            let mut aligned = true;
            let mut padding = 0;

            for (i, child) in node.children().enumerate() {
                let node = self.tree.node(child);

                if node.tag() == close_tag {
                    if multiline {
                        self.indent_decrease();
                    }
                    self.emit_token(node);
                } else if node.tag() == open_tag {
                    self.emit_token(node);
                    if multiline {
                        self.indent_increase();
                    }
                } else {
                    self.output.reserve(padding);
                    for _ in 0..padding {
                        self.output.push(' ');
                    }

                    let len_before = self.output.len();
                    let line_before = self.state.line;

                    self.emit_node(child);

                    let len_after = self.output.len();
                    let line_after = self.state.line;

                    if multiline && multicolumn {
                        if line_before != line_after {
                            multicolumn = false;
                            self.state = state.clone();
                            self.output.truncate(output_len);
                            continue 'outer;
                        }

                        let size = self.output[len_before..len_after].trim().len();
                        if column >= column_sizes.len() {
                            column_sizes.push(size);
                        } else if size > column_sizes[column] {
                            column_sizes[column] = size;
                            aligned = false;
                        } else {
                            // +1 for spacing between arguments
                            padding = column_sizes[column] - size + 1;
                        }
                        column += 1;
                    }
                }

                if multiline && i < count - 1 {
                    let newline_min = if multicolumn { 0 } else { 1 };
                    if self.emit_newlines(newline_min, 1) != 0 {
                        column = 0;
                        padding = 0;
                    }
                }
            }

            if multiline && multicolumn {
                if aligned {
                    break;
                }

                if retries == 0 {
                    multicolumn = false;
                } else {
                    retries -= 1;
                }
                self.state = state.clone();
                self.output.truncate(output_len);

                continue;
            }

            break;
        }
    }

    fn emit_token(&mut self, node: parse::Node) {
        let always_space_before = const {
            parse::EXPRESSION_INFIX_OPS.union(parse::ASSIGNMENT_OPS).with_many(&[
                Tag::Identifier,
                Tag::ThinArrowRight,
                Tag::LCurly,
                Tag::AtSign,
                Tag::KeywordElse,
            ])
        };

        let always_space_after = const {
            parse::EXPRESSION_INFIX_OPS
                .union(parse::ASSIGNMENT_OPS)
                .union(Tag::NUMBERS)
                .union(Tag::KEYWORDS)
                .with_many(&[Tag::Identifier, Tag::ThinArrowRight, Tag::Comma, Tag::Colon])
        };

        const NEVER_SPACE_BEFORE_TERMINATORS: TokenSet = const {
            TokenSet::new(&[
                Tag::RParen,
                Tag::RBracket,
                Tag::TemplateListEnd,
                Tag::Dot,
                Tag::Comma,
                Tag::Colon,
                Tag::SemiColon,
                Tag::PlusPlus,
                Tag::MinusMinus,
            ])
        };
        let never_space_before = const {
            NEVER_SPACE_BEFORE_TERMINATORS.with_many(&[
                Tag::LParen,
                Tag::LBracket,
                Tag::TemplateListStart,
            ])
        };

        let never_space_after = const {
            TokenSet::new(&[
                Tag::LParen,
                Tag::LBracket,
                Tag::TemplateListStart,
                Tag::Dot,
                Tag::AtSign,
            ])
        };

        let never = || {
            never_space_before.contains(node.tag())
                || never_space_after.contains(self.state.previous_token)
        };

        let always = || {
            always_space_before.contains(node.tag())
                || always_space_after.contains(self.state.previous_token)
        };

        let exception = || match (self.state.previous_token, node.tag()) {
            (keyword, Tag::TemplateListStart) if keyword.is_keyword() => false,
            (keyword, after)
                if keyword.is_keyword() && NEVER_SPACE_BEFORE_TERMINATORS.contains(after) =>
            {
                false
            },
            (keyword, _) if keyword.is_keyword() => true,
            (infix_operator, Tag::LParen)
                if parse::EXPRESSION_INFIX_OPS.contains(infix_operator) =>
            {
                true
            },
            _ => false,
        };

        if self.state.force_no_space_before {
            self.state.force_no_space_before = false;
        } else if (!never() && always()) || exception() {
            self.emit_space();
        }

        self.emit_range(node.byte_range(), 0);

        self.state.previous_token = node.tag();
    }

    fn emit_comments_before(&mut self, offset: usize, max_lines_after: usize) -> Option<usize> {
        let mut lines = None;

        while let Some(extra) = self.tree.extra(self.state.next_extra) {
            let extra_range = extra.byte_range();
            if extra_range.start < offset {
                self.state.next_extra += 1;
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

        self.state.last_emit = range.end;
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

        if self.state.needs_indent {
            for _ in 0..self.state.indent {
                self.output.push(' ');
            }
            self.state.needs_indent = false;
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
        let remainder = &self.source[self.state.last_emit..];
        let trimmed = remainder.trim_start();
        let whitespace = &remainder[..remainder.len() - trimmed.len()];
        let mut count = whitespace.chars().filter(|x| *x == '\n').count();

        self.state.last_emit += whitespace.len();

        let mut emitted = 0;
        if count == 0 && min != 0 {
            if let Some(lines_after) = self.emit_comments_before(self.state.last_emit + 1, max) {
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
        self.state.needs_indent = true;
        self.state.line += 1;
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
                    if (1 + 1 == 2) { return vec4(); } else { return vec4(1.0); }

                    // multi-line if-else-statement
                    if (1 + 1 == 2) {
                        return vec4();
                    } else if (false) {
                        return vec4(0.5);
                    } else {
                        return vec4(1.0);
                    }
                }

                struct Uniforms {
                    x: f32,
                    y: f32 // missing commas should be inserted

                    // multiline type specifiers are allowed if you really want
                    complex_array: array<vec2<f32>, 1234,>
                }


                @group(0) @binding(0)
                var<uniform> uniforms: Uniforms;

                fn loops() {
                    for (var i = 0; i < 10; i++) {
                        for (i = 2; i < 28; i += 3) {
                            while i < 3 {
                                if i == 2 {
                                    return arr[i];
                                }
                            }
                        }
                    }
                }

                // we should preserve line breaks in math
                const x = 1 + 2
                    + 4
                    + 8 / some_call_to_a_multiline_function(1, 2, 3,)
                    + 15 * -4
                    - 14 >> (1 + 2);
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
                    if (1 + 1 == 2) { return vec4(); } else { return vec4(1.0); }

                    // multi-line if-else-statement
                    if (1 + 1 == 2) {
                        return vec4();
                    } else if (false) {
                        return vec4(0.5);
                    } else {
                        return vec4(1.0);
                    }
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

                fn loops() {
                    for (var i = 0; i < 10; i++) {
                        for (i = 2; i < 28; i += 3) {
                            while i < 3 {
                                if i == 2 {
                                    return arr[i];
                                }
                            }
                        }
                    }
                }

                // we should preserve line breaks in math
                const x = 1 + 2
                    + 4
                    + 8 / some_call_to_a_multiline_function(
                        1,
                        2,
                        3,
                    )
                    + 15 * -4
                    - 14 >> (1 + 2);
            "#]],
        )
    }

    #[test]
    fn wrapping() {
        check_formatting(
            indoc! {r#"
                fn main() {
                    let wrap_none = mat2x2(1,2,3,4);
                    let wrap_auto = mat2x2(1,2,3,4,);
                    let wrap_keep = mat2x2(
                        1,2,
                        3,4,
                    );
                }
            "#},
            expect![[r#"
                fn main() {
                    let wrap_none = mat2x2(1, 2, 3, 4);
                    let wrap_auto = mat2x2(
                        1,
                        2,
                        3,
                        4,
                    );
                    let wrap_keep = mat2x2(
                        1, 2,
                        3, 4,
                    );
                }
            "#]],
        )
    }

    #[test]
    fn alignment() {
        check_formatting(
            indoc! {r#"
                fn main() {
                    let align_simple = mat2x2(
                        123,4,5,
                        6,78,90,
                    );
                    let align_keep_complex = mat2x2(
                        1,vec2(a,b),2,
                        3456,78,90,
                    );
                    let align_skip_complex = mat2x2(
                        123,vec2(a,b,),4,
                        6,78,90,
                    );
                }
            "#},
            expect![[r#"
                fn main() {
                    let align_simple = mat2x2(
                        123, 4,  5,
                        6,   78, 90,
                    );
                    let align_keep_complex = mat2x2(
                        1,    vec2(a, b), 2,
                        3456, 78,         90,
                    );
                    let align_skip_complex = mat2x2(
                        123,
                        vec2(
                            a,
                            b,
                        ),
                        4,
                        6,
                        78,
                        90,
                    );
                }
            "#]],
        )
    }
}
