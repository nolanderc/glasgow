use crate::util::U24;

use super::{Node, Tag};

pub struct Tokenizer<'src> {
    source: &'src str,
    offset: usize,
    template_starts: TemplateOffsetIter,
    template_ends: TemplateOffsetIter,
}

type TemplateOffsetIter = std::iter::Peekable<std::vec::IntoIter<u32>>;

impl<'src> Tokenizer<'src> {
    pub fn new(source: &'src str) -> Tokenizer<'src> {
        assert!(
            u32::try_from(source.len()).is_ok(),
            "source code is larger than what can be represented with 32-bit integer"
        );

        let templates = TemplateLists::extract(source);

        Tokenizer {
            source,
            offset: 0,
            template_starts: templates.starts.into_iter().peekable(),
            template_ends: templates.ends.into_iter().peekable(),
        }
    }

    pub fn empty() -> Tokenizer<'static> {
        Tokenizer {
            source: "",
            offset: 0,
            template_starts: Vec::new().into_iter().peekable(),
            template_ends: Vec::new().into_iter().peekable(),
        }
    }

    pub fn next(&mut self) -> Node {
        self.offset += whitespace(&self.source[self.offset..]);
        let (token, length) = self.strip_token();
        let node = Node {
            tag: token,
            length: U24::from_usize(length).expect("token exceeds maximum length"),
            offset: self.offset as u32,
        };
        self.offset += length;
        node
    }

    fn strip_token(&mut self) -> (Tag, usize) {
        let source = &self.source[self.offset..];

        if source.is_empty() {
            return (Tag::Eof, 0);
        }

        let bytes = source.as_bytes();
        match bytes[0] {
            // catch regular ASCII identifiers here
            b'_' | b'a'..=b'z' | b'A'..=b'Z' => {
                let len = identifier(source);
                (classify_identifier(&source[..len]), len)
            },

            // zero or hexadecimal number
            b'0' => number_prefix_0(source),

            // decimal number
            b'1'..=b'9' => number_prefix_1_9(source),

            // template list start or comparison/shift
            b'<' => {
                if self.template_starts.peek() == Some(&(self.offset as u32)) {
                    self.template_starts.next();
                    return (Tag::TemplateListStart, 1);
                }

                match bytes.get(1) {
                    Some(b'<') => match bytes.get(2) {
                        Some(b'=') => (Tag::LessLessEqual, 3),
                        _ => (Tag::LessLess, 2),
                    },
                    Some(b'=') => (Tag::LessEqual, 2),
                    _ => (Tag::Less, 1),
                }
            },

            // template list end or comparison/shift
            b'>' => {
                if self.template_ends.peek() == Some(&(self.offset as u32)) {
                    self.template_ends.next();
                    return (Tag::TemplateListEnd, 1);
                }

                match bytes.get(1) {
                    Some(b'>') => match bytes.get(2) {
                        Some(b'=') => (Tag::GreaterGreaterEqual, 3),
                        _ => (Tag::GreaterGreater, 2),
                    },
                    Some(b'=') => (Tag::GreaterEqual, 2),
                    _ => (Tag::Greater, 1),
                }
            },

            b'&' => match bytes.get(1) {
                Some(b'&') => (Tag::AmpersandAmpersand, 2),
                Some(b'=') => (Tag::AmpersandEqual, 2),
                _ => (Tag::Ampersand, 1),
            },
            b'@' => (Tag::AtSign, 1),
            b'/' => match bytes.get(1) {
                Some(b'=') => (Tag::SlashEqual, 2),
                Some(b'/' | b'*') => (Tag::Comment, comment(source)),
                _ => (Tag::Slash, 1),
            },
            b'!' => match bytes.get(1) {
                Some(b'=') => (Tag::ExclamationEqual, 2),
                _ => (Tag::Exclamation, 1),
            },
            b'[' => (Tag::LBracket, 1),
            b']' => (Tag::RBracket, 1),
            b'{' => (Tag::LCurly, 1),
            b'}' => (Tag::RCurly, 1),
            b':' => (Tag::Colon, 1),
            b',' => (Tag::Comma, 1),
            b'=' => match bytes.get(1) {
                Some(b'=') => (Tag::EqualEqual, 2),
                _ => (Tag::Equal, 1),
            },
            b'%' => match bytes.get(1) {
                Some(b'=') => (Tag::PercentEqual, 2),
                _ => (Tag::Percent, 1),
            },
            b'-' => match bytes.get(1) {
                Some(b'>') => (Tag::ThinArrowRight, 2),
                Some(b'-') => (Tag::MinusMinus, 2),
                Some(b'=') => (Tag::MinusEqual, 2),
                _ => (Tag::Minus, 1),
            },
            b'.' => (Tag::Dot, 1),
            b'+' => match bytes.get(1) {
                Some(b'+') => (Tag::PlusPlus, 2),
                Some(b'=') => (Tag::PlusEqual, 2),
                _ => (Tag::Plus, 1),
            },
            b'|' => match bytes.get(1) {
                Some(b'|') => (Tag::BarBar, 2),
                Some(b'=') => (Tag::BarEqual, 2),
                _ => (Tag::Bar, 1),
            },
            b'(' => (Tag::LParen, 1),
            b')' => (Tag::RParen, 1),
            b';' => (Tag::SemiColon, 1),
            b'*' => match bytes.get(1) {
                Some(b'=') => (Tag::AsteriskEqual, 2),
                _ => (Tag::Asterisk, 1),
            },
            b'~' => (Tag::Tilde, 1),
            b'^' => match bytes.get(1) {
                Some(b'=') => (Tag::ChevronEqual, 2),
                _ => (Tag::Chevron, 1),
            },

            byte => {
                // catch more complex identifiers down here
                let len = identifier(source);
                if len != 0 {
                    return (classify_identifier(&source[..len]), len);
                }
                (Tag::InvalidToken, utf8_codepoint_length(byte).into())
            },
        }
    }
}

#[derive(Debug)]
struct TemplateLists {
    starts: Vec<u32>,
    ends: Vec<u32>,
}

impl TemplateLists {
    pub fn extract(source: &str) -> Self {
        let skip_whitespace = |offset: &mut usize| {
            let mut i = *offset;
            loop {
                i += whitespace(&source[i..]);
                if i >= source.len() {
                    break;
                }

                let comment = comment(&source[i..]);
                if comment != 0 {
                    i += comment;
                    continue;
                }

                *offset = i;
                break;
            }
        };

        struct Pending {
            depth: u32,
            start: u32,
        }

        let mut pending = Vec::<Pending>::new();
        let mut starts = Vec::new();
        let mut ends = Vec::new();

        let mut depth = 0;
        let mut offset = 0;

        assert_eq!(
            source.len() as u32 as usize,
            source.len(),
            "source code length must fit in a u32"
        );

        loop {
            skip_whitespace(&mut offset);
            if offset >= source.len() {
                break;
            }

            match source.as_bytes()[offset] {
                b'>' => {
                    if let Some(last) = pending.last() {
                        if last.depth == depth {
                            starts.push(last.start);
                            ends.push(offset as u32);
                            pending.pop();
                            offset += 1;
                            continue;
                        }
                    }

                    offset += 1;

                    if source.as_bytes().get(offset) == Some(&b'=') {
                        offset += 1;
                    }
                },
                b'(' | b'[' => {
                    depth += 1;
                    offset += 1
                },
                b')' | b']' => {
                    while matches!(pending.last(), Some(x) if x.depth >= depth) {
                        pending.pop();
                    }
                    depth = depth.saturating_sub(1);
                    offset += 1
                },

                b'!' => {
                    offset += 1;
                    if source.as_bytes().get(offset) == Some(&b'=') {
                        offset += 1;
                    }
                },

                b'=' => {
                    offset += 1;
                    if source.as_bytes().get(offset) == Some(&b'=') {
                        offset += 1;
                        continue;
                    }

                    // we are in an assignment
                    depth = 0;
                    pending.clear();
                },

                // cannot appear in an expression:
                b';' | b'{' | b':' => {
                    offset += 1;
                    depth = 0;
                    pending.clear();
                },

                byte => {
                    let ident_length = identifier(&source[offset..]);
                    if ident_length != 0 {
                        offset += ident_length;
                        skip_whitespace(&mut offset);
                        if !matches!(source.as_bytes().get(offset), Some(b'<')) {
                            continue;
                        }
                        offset += 1;
                        if matches!(source.as_bytes().get(offset), Some(b'<' | b'=')) {
                            offset += 1;
                            continue;
                        }
                        pending.push(Pending { depth, start: offset as u32 - 1 });
                        continue;
                    }

                    let number_length = number_conservative(&source[offset..]);
                    if number_length != 0 {
                        offset += number_length;
                        continue;
                    }

                    if source[offset..].starts_with("&&") || source[offset..].starts_with("||") {
                        while matches!(pending.last(), Some(x) if x.depth >= depth) {
                            pending.pop();
                        }
                        offset += 2;
                        continue;
                    }

                    // We already know this is valid UTF-8, so we don't bother checking for
                    // invalid patterns.
                    #[allow(clippy::match_overlapping_arm)]
                    let codepoint_len = match byte {
                        ..=0b0111_1111 => 1,
                        ..=0b1101_1111 => 2,
                        ..=0b1110_1111 => 3,
                        ..=0b1111_1111 => 4,
                    };
                    offset += codepoint_len;
                },
            }
        }

        starts.sort_unstable();
        assert!(starts.windows(2).all(|x| x[0] < x[1]));
        assert!(ends.windows(2).all(|x| x[0] < x[1]));

        TemplateLists { starts, ends }
    }

    #[cfg(test)]
    fn pairs(&self) -> Vec<(u32, u32)> {
        let starts = self.starts.as_slice();
        let ends = self.ends.as_slice();

        let mut pairs = Vec::new();
        let mut unmatched = Vec::new();

        let mut start_index = 0;
        let mut end_index = 0;
        while start_index < starts.len() {
            let curr = starts[start_index];

            let end = ends[end_index];
            if end < curr {
                let prev = unmatched.pop().unwrap();
                pairs.push((prev, end));
                end_index += 1;
                continue;
            }

            if let Some(&next) = starts.get(start_index + 1) {
                if next < end {
                    unmatched.push(curr);
                    start_index += 1;
                    continue;
                }
            }

            pairs.push((curr, end));
            start_index += 1;
            end_index += 1;
        }

        while let Some(start) = unmatched.pop() {
            let end = ends[end_index];
            pairs.push((start, end));
            end_index += 1;
        }

        assert_eq!(start_index, starts.len());
        assert_eq!(end_index, ends.len());

        pairs
    }
}

/// Assuming we have the first byte of a valid UTF-8 encoded codepoint, returns its length.
fn utf8_codepoint_length(first_byte: u8) -> u8 {
    #[allow(clippy::match_overlapping_arm)]
    match first_byte {
        ..=0b0111_1111 => 1,
        ..=0b1101_1111 => 2,
        ..=0b1110_1111 => 3,
        ..=0b1111_1111 => 4,
    }
}

fn identifier(source: &str) -> usize {
    let mut chars = source.chars();
    let Some(first) = chars.next() else { return 0 };
    if !(first == '_' || first.is_alphabetic()) {
        return 0;
    }

    let mut rest = chars.as_str().len();
    while let Some(x) = chars.next() {
        if x == '_' || x.is_alphanumeric() {
            rest = chars.as_str().len();
        } else {
            break;
        }
    }
    source.len() - rest
}

fn classify_identifier(identifier: &str) -> Tag {
    match identifier {
        "alias" => Tag::KeywordAlias,
        "break" => Tag::KeywordBreak,
        "case" => Tag::KeywordCase,
        "const" => Tag::KeywordConst,
        "const_assert" => Tag::KeywordConstAssert,
        "continue" => Tag::KeywordContinue,
        "continuing" => Tag::KeywordContinuing,
        "default" => Tag::KeywordDefault,
        "diagnostic" => Tag::KeywordDiagnostic,
        "discard" => Tag::KeywordDiscard,
        "else" => Tag::KeywordElse,
        "enable" => Tag::KeywordEnable,
        "false" => Tag::KeywordFalse,
        "fn" => Tag::KeywordFn,
        "for" => Tag::KeywordFor,
        "if" => Tag::KeywordIf,
        "let" => Tag::KeywordLet,
        "loop" => Tag::KeywordLoop,
        "override" => Tag::KeywordOverride,
        "requires" => Tag::KeywordRequires,
        "return" => Tag::KeywordReturn,
        "struct" => Tag::KeywordStruct,
        "switch" => Tag::KeywordSwitch,
        "true" => Tag::KeywordTrue,
        "var" => Tag::KeywordVar,
        "while" => Tag::KeywordWhile,

        _ => Tag::Identifier,
    }
}

fn number_prefix_0(source: &str) -> (Tag, usize) {
    let bytes = source.as_bytes();
    debug_assert_eq!(bytes[0], b'0');

    match bytes.get(1) {
        Some(b'x' | b'X') => {
            //   0[xX][0-9a-fA-F]+[iu]?
            //   0[xX][0-9a-fA-F]*\.[0-9a-fA-F]+([pP][+-]?[0-9]+[fh]?)?
            //   0[xX][0-9a-fA-F]+\.[0-9a-fA-F]*([pP][+-]?[0-9]+[fh]?)?
            //   0[xX][0-9a-fA-F]+[pP][+-]?[0-9]+[fh]?
            let mut i = 2;
            let mut integer = true;

            while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                i += 1;
            }

            // fraction
            if matches!(bytes.get(i), Some(b'.')) {
                i += 1;
                integer = false;
                while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                    i += 1;
                }
            }

            // exponent
            if matches!(bytes.get(i), Some(b'p' | b'P')) {
                i += 1;
                integer = false;

                if matches!(bytes.get(i), Some(b'+' | b'-')) {
                    i += 1;
                }

                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }

            // type suffix
            if matches!(bytes.get(i), Some(b'f' | b'h')) {
                i += 1;
                integer = false;
            }
            if integer && matches!(bytes.get(i), Some(b'i' | b'u')) {
                i += 1;
            }

            (if integer { Tag::IntegerHex } else { Tag::FloatHex }, i)
        },
        _ => number_prefix_1_9(source),
    }
}

fn number_prefix_1_9(source: &str) -> (Tag, usize) {
    //   [1-9][0-9]*[iu]?
    //   [1-9][0-9]*[fh]
    //   [0-9]*\.[0-9]+([eE][+-]?[0-9]+)?[fh]?
    //   [0-9]+\.[0-9]*([eE][+-]?[0-9]+)?[fh]?
    //   [0-9]+[eE][+-]?[0-9]+[fh]?
    let mut i = 1;
    let mut integer = true;

    let bytes = source.as_bytes();

    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }

    // fraction
    if matches!(bytes.get(i), Some(b'.')) {
        i += 1;
        integer = false;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }

    // exponent
    if matches!(bytes.get(i), Some(b'e' | b'E')) {
        i += 1;
        integer = false;

        if matches!(bytes.get(i), Some(b'+' | b'-')) {
            i += 1;
        }

        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }

    // type suffix
    if matches!(bytes.get(i), Some(b'f' | b'h')) {
        i += 1;
        integer = false;
    }
    if integer && matches!(bytes.get(i), Some(b'i' | b'u')) {
        i += 1;
    }

    (if integer { Tag::IntegerDecimal } else { Tag::FloatDecimal }, i)
}

/// Matches anything that has the general shape of a number: it starts with a digit followed by
/// zero or more digits, letters (for exponents and precise typed literals), and pluses or minuses
/// (for exponents).
fn number_conservative(source: &str) -> usize {
    if source.is_empty() {
        return 0;
    }

    let bytes = source.as_bytes();
    if !bytes[0].is_ascii_digit() {
        return 0;
    }

    let mut i = 1;
    while i < bytes.len()
        && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'+' | b'-' | b'.'))
    {
        i += 1
    }

    i
}

fn comment(source: &str) -> usize {
    let mut i = 0;
    if source.starts_with("//") {
        i += 2;
        let bytes = source.as_bytes();
        while i < bytes.len() && bytes[i] != b'\n' {
            i += 1;
        }
    } else if source.starts_with("/*") {
        i += 2;
        let mut depth = 1;
        while i < source.len() {
            if source[i..].starts_with("/*") {
                depth += 1;
                i += 2;
            } else if source[i..].starts_with("*/") {
                depth -= 1;
                i += 2;
                if depth == 0 {
                    break;
                }
            } else {
                i += 1;
            }
        }
    }
    i
}

fn whitespace(source: &str) -> usize {
    let mut rest = source.len();
    let mut chars = source.chars();
    while let Some(x) = chars.next() {
        if x.is_whitespace() {
            rest = chars.as_str().len();
        } else {
            break;
        }
    }
    source.len() - rest
}

#[derive(Clone, Copy)]
pub struct TokenSet {
    bits: u128,
}

impl TokenSet {
    pub const fn empty() -> TokenSet {
        TokenSet { bits: 0 }
    }

    pub const fn new(tokens: &[Tag]) -> TokenSet {
        Self::empty().with_many(tokens)
    }

    pub const fn with(mut self, token: Tag) -> TokenSet {
        self.bits |= Self::mask(token);
        self
    }

    pub const fn with_many(mut self, mut tokens: &[Tag]) -> TokenSet {
        while let Some((token, rest)) = tokens.split_first() {
            tokens = rest;
            self.bits |= Self::mask(*token);
        }
        self
    }

    pub const fn contains(self, token: Tag) -> bool {
        self.bits & Self::mask(token) != 0
    }

    pub const fn union(self, other: TokenSet) -> TokenSet {
        TokenSet { bits: self.bits | other.bits }
    }

    const fn mask(token: Tag) -> u128 {
        assert!(token.is_token() && (token as u8) < 128);
        1u128 << token as u8
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::*;

    fn check_template_lists(source: &str, expected: expect_test::Expect) {
        let templates = TemplateLists::extract(source);

        let mut output = String::new();
        for (start, end) in templates.pairs() {
            output += &format!("{start}:{end} = {:?}\n", &source[start as usize..=end as usize]);
        }

        expected.assert_eq(&output);
    }

    #[test]
    fn template_lists_vec4() {
        check_template_lists(
            "vec4<f32>",
            expect![[r#"
                4:8 = "<f32>"
            "#]],
        );
    }

    #[test]
    fn template_lists_nested() {
        check_template_lists(
            "array<array<tuple<vec4<f32>, mat2x2<bool>>>, 3>",
            expect![[r#"
                22:26 = "<f32>"
                35:40 = "<bool>"
                17:41 = "<vec4<f32>, mat2x2<bool>>"
                11:42 = "<tuple<vec4<f32>, mat2x2<bool>>>"
                5:46 = "<array<tuple<vec4<f32>, mat2x2<bool>>>, 3>"
            "#]],
        );
    }

    #[test]
    fn template_lists_parens_insignificant() {
        check_template_lists(
            "A ( B < C, D > ( E ) )",
            expect![[r#"
                6:13 = "< C, D >"
            "#]],
        );
    }

    #[test]
    fn template_lists_parens_significant() {
        check_template_lists(
            "array<i32,select(2,3,a>b)>",
            expect![[r#"
                5:25 = "<i32,select(2,3,a>b)>"
            "#]],
        );
    }

    #[test]
    fn template_lists_across_brackets() {
        check_template_lists("a[b<d]>()", expect![""]);
    }

    #[test]
    fn template_lists_comparison() {
        check_template_lists(
            "A<B<=C>",
            expect![[r#"
                1:6 = "<B<=C>"
            "#]],
        );
    }

    #[test]
    fn template_lists_comparison_less() {
        check_template_lists(
            "A<B<=C>",
            expect![[r#"
                1:6 = "<B<=C>"
            "#]],
        );
    }

    #[test]
    fn template_lists_comparison_greater() {
        check_template_lists(
            "A<(B>=C)>",
            expect![[r#"
                1:8 = "<(B>=C)>"
            "#]],
        );
    }

    #[test]
    fn template_lists_comparison_not_equal() {
        check_template_lists(
            "A<(B!=C)>",
            expect![[r#"
                1:8 = "<(B!=C)>"
            "#]],
        );
    }

    #[test]
    fn template_lists_comparison_equal() {
        check_template_lists(
            "A<(B==C)>",
            expect![[r#"
                1:8 = "<(B==C)>"
            "#]],
        );
    }

    #[test]
    fn template_lists_literal() {
        check_template_lists("1.0f<f32>", expect![[]]);
    }

    #[test]
    fn template_lists_short_circuit() {
        check_template_lists("a < b || c > d", expect![[]]);
    }

    #[test]
    fn template_lists_shifting() {
        check_template_lists("a << 1 >> 3", expect![[]]);
    }

    #[test]
    fn template_lists_call() {
        // this is weird, but intended, the user can always disambiguate with parenthesis.
        // See section "Breakages": https://github.com/gpuweb/gpuweb/issues/3770#issue-1553540805
        check_template_lists(
            "foo(a < b, c > d)",
            expect![[r#"
                6:13 = "< b, c >"
            "#]],
        );
        check_template_lists("foo((a < b), (c > d))", expect![[]]);
    }
}
