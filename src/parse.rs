pub mod token;

use crate::util::U24;

use self::token::TokenSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Tag {
    // vvvvv tokens vvvvv //
    /// A virtual token signifying the end of file. Only used during parsing.
    Eof = 0,

    /// An invalid token which could not be classified.
    InvalidToken,

    /// A comment.
    Comment,

    /// A preprocessor directive
    Preprocessor,

    Identifier,

    IntegerDecimal,
    IntegerHex,
    FloatDecimal,
    FloatHex,

    KeywordAlias,
    KeywordBreak,
    KeywordCase,
    KeywordConst,
    KeywordConstAssert,
    KeywordContinue,
    KeywordContinuing,
    KeywordDefault,
    KeywordDiagnostic,
    KeywordDiscard,
    KeywordElse,
    KeywordEnable,
    KeywordFalse,
    KeywordFn,
    KeywordFor,
    KeywordIf,
    KeywordLet,
    KeywordLoop,
    KeywordOverride,
    KeywordRequires,
    KeywordReturn,
    KeywordStruct,
    KeywordSwitch,
    KeywordTrue,
    KeywordVar,
    KeywordWhile,

    /// `@`
    AtSign,

    /// `->`
    ThinArrowRight,

    /// `(`
    LParen,
    /// `)`
    RParen,

    /// `[`
    LBracket,
    /// `]`
    RBracket,

    /// `{`
    LCurly,
    /// `}`
    RCurly,

    /// `.`
    Dot,
    /// `,`
    Comma,
    /// `:`
    Colon,
    /// `;`
    SemiColon,

    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Asterisk,
    /// `/`
    Slash,
    /// `%`
    Percent,
    /// `^`
    Chevron,
    /// `~`
    Tilde,

    /// `+=`
    PlusEqual,
    /// `-=`
    MinusEqual,
    /// `*=`
    AsteriskEqual,
    /// `/=`
    SlashEqual,
    /// `%=`
    PercentEqual,
    /// `^=`
    ChevronEqual,
    /// `&=`
    AmpersandEqual,
    /// `|=`
    BarEqual,

    /// `++`
    PlusPlus,
    /// `--`
    MinusMinus,

    /// `<`
    Less,
    /// `<=`
    LessEqual,
    /// `<<`
    LessLess,
    /// `<<=`
    LessLessEqual,

    /// `>`
    Greater,
    /// `>=`
    GreaterEqual,
    /// `>>`
    GreaterGreater,
    /// `>>=`
    GreaterGreaterEqual,

    /// `=`
    Equal,
    /// `==`
    EqualEqual,

    /// `!`
    Exclamation,
    /// `!=`
    ExclamationEqual,

    /// `&`
    Ampersand,
    /// `&&`
    AmpersandAmpersand,

    /// `|`
    Bar,
    /// `||`
    BarBar,

    /// `<`
    TemplateListStart,
    /// `>`
    TemplateListEnd,
    // ^^^^^ tokens ^^^^^ //
    // vvvvv syntax vvvvv //
    InvalidSyntax,

    Root,

    Argument,
    ArgumentList,
    Attribute,
    AttributeList,
    DeclAlias,
    DeclConst,
    DeclConstAssert,
    DeclFn,
    DeclFnOutput,
    DeclFnParameter,
    DeclFnParameterList,
    DeclOverride,
    DeclStruct,
    DeclStructField,
    DeclStructFieldList,
    DeclVar,
    DirectiveDiagnostic,
    DirectiveDiagnosticName,
    DirectiveEnable,
    DirectiveEnableName,
    DirectiveRequires,
    DirectiveRequiresName,
    ExprCall,
    ExprIndex,
    ExprInfix,
    ExprMember,
    ExprParens,
    ExprPrefix,
    IdentifierWithTemplate,
    StmtAssign,
    StmtBlock,
    StmtBreak,
    StmtContinue,
    StmtContinuing,
    StmtDecrement,
    StmtDiscard,
    StmtExpr,
    StmtFor,
    StmtIf,
    StmtIfBranch,
    StmtIncrement,
    StmtLet,
    StmtLoop,
    StmtReturn,
    StmtSwitch,
    StmtSwitchBranch,
    StmtSwitchBranchList,
    StmtSwitchBranchCase,
    StmtWhile,
    TemplateList,
    TemplateParameter,
    // ^^^^^ syntax ^^^^^ //
}

impl Tag {
    pub const fn is_token(self) -> bool {
        (self as u8) < (Tag::InvalidSyntax as u8)
    }

    pub const fn is_syntax(self) -> bool {
        !self.is_token()
    }

    pub const NUMBERS: TokenSet =
        TokenSet::new(&[Tag::IntegerDecimal, Tag::IntegerHex, Tag::FloatDecimal, Tag::FloatHex]);

    pub const KEYWORDS: TokenSet = TokenSet::new(&[
        Tag::KeywordAlias,
        Tag::KeywordBreak,
        Tag::KeywordCase,
        Tag::KeywordConst,
        Tag::KeywordConstAssert,
        Tag::KeywordContinue,
        Tag::KeywordContinuing,
        Tag::KeywordDefault,
        Tag::KeywordDiagnostic,
        Tag::KeywordDiscard,
        Tag::KeywordElse,
        Tag::KeywordEnable,
        Tag::KeywordFalse,
        Tag::KeywordFn,
        Tag::KeywordFor,
        Tag::KeywordIf,
        Tag::KeywordLet,
        Tag::KeywordLoop,
        Tag::KeywordOverride,
        Tag::KeywordRequires,
        Tag::KeywordReturn,
        Tag::KeywordStruct,
        Tag::KeywordSwitch,
        Tag::KeywordTrue,
        Tag::KeywordVar,
        Tag::KeywordWhile,
    ]);

    pub const fn is_keyword(self) -> bool {
        Self::KEYWORDS.contains(self)
    }

    pub const fn token_description(self) -> Option<&'static str> {
        Some(match self {
            Tag::Eof => "the end of file",
            Tag::InvalidToken => "an invalid token",
            Tag::Comment => "a comment",
            Tag::Preprocessor => "a preprocessor directive",
            Tag::Identifier => "an identifier",
            Tag::IntegerDecimal => "an integer literal",
            Tag::IntegerHex => "an integer literal",
            Tag::FloatDecimal => "a floating point literal",
            Tag::FloatHex => "a floating point literal",
            Tag::KeywordAlias => "`alias`",
            Tag::KeywordBreak => "`break`",
            Tag::KeywordCase => "`case`",
            Tag::KeywordConst => "`const`",
            Tag::KeywordConstAssert => "`const_assert`",
            Tag::KeywordContinue => "`continue`",
            Tag::KeywordContinuing => "`continuing`",
            Tag::KeywordDefault => "`default`",
            Tag::KeywordDiagnostic => "`diagnostic`",
            Tag::KeywordDiscard => "`discard`",
            Tag::KeywordElse => "`else`",
            Tag::KeywordEnable => "`enable`",
            Tag::KeywordFalse => "`false`",
            Tag::KeywordFn => "`fn`",
            Tag::KeywordFor => "`for`",
            Tag::KeywordIf => "`if`",
            Tag::KeywordLet => "`let`",
            Tag::KeywordLoop => "`loop`",
            Tag::KeywordOverride => "`override`",
            Tag::KeywordRequires => "`requires`",
            Tag::KeywordReturn => "`return`",
            Tag::KeywordStruct => "`struct`",
            Tag::KeywordSwitch => "`switch`",
            Tag::KeywordTrue => "`true`",
            Tag::KeywordVar => "`var`",
            Tag::KeywordWhile => "`while`",
            Tag::AtSign => "`@`",
            Tag::ThinArrowRight => "`->`",
            Tag::LParen => "`(`",
            Tag::RParen => "`)`",
            Tag::LBracket => "`[`",
            Tag::RBracket => "`]`",
            Tag::LCurly => "`{`",
            Tag::RCurly => "`}`",
            Tag::Dot => "`.`",
            Tag::Comma => "`,`",
            Tag::Colon => "`:`",
            Tag::SemiColon => "`;`",
            Tag::Plus => "`+`",
            Tag::Minus => "`-`",
            Tag::Asterisk => "`*`",
            Tag::Slash => "`/`",
            Tag::Percent => "`%`",
            Tag::Chevron => "`^`",
            Tag::Tilde => "`~`",
            Tag::PlusEqual => "`+=`",
            Tag::MinusEqual => "`-=`",
            Tag::AsteriskEqual => "`*=`",
            Tag::SlashEqual => "`/=`",
            Tag::PercentEqual => "`%=`",
            Tag::ChevronEqual => "`^+`",
            Tag::AmpersandEqual => "`&=`",
            Tag::BarEqual => "`|=`",
            Tag::PlusPlus => "`++`",
            Tag::MinusMinus => "`--`",
            Tag::Less => "`<`",
            Tag::LessEqual => "`<=`",
            Tag::LessLess => "`<<`",
            Tag::LessLessEqual => "`<<=`",
            Tag::Greater => "`>`",
            Tag::GreaterEqual => "`>=`",
            Tag::GreaterGreater => "`>>`",
            Tag::GreaterGreaterEqual => "`>>=`",
            Tag::Equal => "`=`",
            Tag::EqualEqual => "`==`",
            Tag::Exclamation => "`!`",
            Tag::ExclamationEqual => "`!=`",
            Tag::Ampersand => "`&`",
            Tag::AmpersandAmpersand => "`i&&`",
            Tag::Bar => "`|`",
            Tag::BarBar => "`||`",
            Tag::TemplateListStart => "`<`",
            Tag::TemplateListEnd => "`>`",

            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Node {
    tag: Tag,
    length: U24,
    offset: u32,
}

impl Node {
    pub fn tag(self) -> Tag {
        self.tag
    }

    pub fn is_token(self) -> bool {
        self.tag.is_token()
    }

    pub fn is_syntax(self) -> bool {
        self.tag.is_syntax()
    }

    pub fn children(
        self,
    ) -> impl ExactSizeIterator<Item = NodeIndex> + DoubleEndedIterator + Clone {
        assert!(self.is_syntax());
        (self.offset..self.offset + self.length.to_u32()).map(NodeIndex)
    }

    fn children_range(self) -> std::ops::Range<usize> {
        assert!(self.is_syntax());
        let offset = self.offset as usize;
        offset..offset + self.length.to_usize()
    }

    pub fn byte_range(self) -> std::ops::Range<usize> {
        assert!(self.is_token());
        let offset = self.offset as usize;
        offset..offset + self.length.to_usize()
    }
}

#[derive(Default)]
pub struct Output {
    pub tree: crate::parse::Tree,
    pub errors: Vec<crate::parse::Error>,
}

#[derive(Default)]
pub struct Tree {
    nodes: Vec<Node>,
    extra: Vec<Node>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeIndex(u32);

impl Tree {
    pub fn node(&self, index: NodeIndex) -> Node {
        self.nodes[index.0 as usize]
    }

    pub fn children(&self, node: Node) -> &[Node] {
        &self.nodes[node.children_range()]
    }

    pub(crate) fn extra(&self, index: usize) -> Option<Node> {
        self.extra.get(index).copied()
    }

    pub fn root_index(&self) -> NodeIndex {
        NodeIndex((self.nodes.len() - 1) as u32)
    }

    pub fn root(&self) -> Node {
        self.node(self.root_index())
    }

    #[inline]
    pub fn byte_range_total(&self, index: NodeIndex) -> Option<std::ops::Range<usize>> {
        let node = self.node(index);
        if node.is_token() {
            return Some(node.byte_range());
        }
        self.byte_range_total_children(node.children())
    }

    pub fn byte_range_total_children(
        &self,
        mut children: impl DoubleEndedIterator<Item = NodeIndex>,
    ) -> Option<std::ops::Range<usize>> {
        let min = loop {
            let child = children.next()?;
            match self.byte_range_total(child) {
                Some(range) => break range,
                None => continue,
            }
        };

        let max = loop {
            let Some(child) = children.next_back() else { break min.clone() };
            match self.byte_range_total(child) {
                Some(range) => break range,
                None => continue,
            }
        };

        Some(min.start..max.end)
    }

    #[cfg(test)]
    pub fn traverse_pre_order<E>(
        &self,
        mut visit: impl FnMut(NodeIndex, Node, usize) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut stack = Vec::with_capacity(32);

        let root = self.root_index().0;
        stack.push(root..root + 1);

        while let Some(top) = stack.last_mut() {
            let Some(curr) = top.next() else {
                stack.pop();
                continue;
            };

            let depth = stack.len() - 1;
            let node = self.node(NodeIndex(curr));
            visit(NodeIndex(curr), node, depth)?;

            if node.tag.is_syntax() {
                let children = node.children_range();
                stack.push(children.start as u32..children.end as u32);
            }
        }

        Ok(())
    }

    pub fn enumerate_token_path_in_range_utf8(
        &self,
        range: std::ops::Range<usize>,
        mut callback: impl FnMut(NodeIndex),
    ) -> bool {
        fn find_in_node(
            nodes: &[Node],
            curr: NodeIndex,
            range: std::ops::Range<usize>,
            callback: &mut impl FnMut(NodeIndex),
        ) -> bool {
            let node = nodes[curr.0 as usize];

            if node.tag.is_token() {
                let node_range = node.byte_range();
                if node_range.start < range.end && range.start < node_range.end {
                    callback(curr as NodeIndex);
                    return true;
                } else {
                    return false;
                }
            }

            let mut candidate_children = node.children_range();

            for child in node.children().rev() {
                let c = nodes[child.0 as usize];
                if c.tag.is_token() {
                    let node_range = c.byte_range();
                    if node_range.end < range.start {
                        candidate_children.start = child.0 as usize + 1;
                        break;
                    }
                    if range.end <= node_range.start {
                        candidate_children.end = child.0 as usize;
                    }
                }
            }

            for child in candidate_children.rev() {
                if find_in_node(nodes, NodeIndex(child as u32), range.clone(), callback) {
                    callback(curr as NodeIndex);
                    return true;
                }
            }

            false
        }

        find_in_node(&self.nodes, self.root_index(), range, &mut callback)
    }

    pub fn token_path_in_range_utf8(&self, range: std::ops::Range<usize>) -> Vec<NodeIndex> {
        let mut path = Vec::new();
        self.enumerate_token_path_in_range_utf8(range, |index| path.push(index));
        path.reverse();
        path
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Error {
    pub token_found: Node,
    pub token_previous: Node,
    pub expected: Expected,
}

impl Error {
    pub fn message(self, source: &str) -> String {
        let found = match self.token_found.tag {
            Tag::Eof => "the end of file".to_string(),
            _ => {
                let snippet = &source[self.token_found.byte_range()];
                let print_safe = snippet.bytes().all(|x| x.is_ascii_graphic() || x == b' ');
                if print_safe {
                    format!("`{snippet}`")
                } else {
                    format!("`{}`", snippet.escape_debug().collect::<String>())
                }
            },
        };

        format!("{}, but found {found}", self.expected.message())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Expected {
    Token(Tag),
    Declaration,
    Expression,
    Statement,
}

impl Expected {
    pub fn message(self) -> String {
        match self {
            Expected::Token(token) => match token.token_description() {
                Some(description) => format!("expected {description}"),
                _ => unreachable!("expected a token, but found {token:?}"),
            },
            Expected::Declaration => "expected a declaration".into(),
            Expected::Expression => "expected an expression".into(),
            Expected::Statement => "expected a statement".into(),
        }
    }
}

pub struct Parser<'src> {
    tokens: token::Tokenizer<'src>,
    current_token: Node,
    previous_token: Node,

    stack: Vec<Node>,
    output: Output,

    fuel: u32,
}

impl<'src> Default for Parser<'src> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'src> Parser<'src> {
    const MAX_FUEL: u32 = 128;

    pub fn new() -> Parser<'static> {
        Parser {
            tokens: token::Tokenizer::empty(),
            current_token: Node { tag: Tag::Eof, length: U24::ZERO, offset: 0 },
            previous_token: Node { tag: Tag::Eof, length: U24::ZERO, offset: 0 },
            stack: Vec::new(),
            output: Output::default(),
            fuel: 0,
        }
    }

    pub fn reset(self, output: Output) -> Parser<'static> {
        Parser { tokens: token::Tokenizer::empty(), output, ..self }
    }

    fn begin(&mut self, source: &'src str) {
        self.tokens = token::Tokenizer::new(source);

        self.stack.clear();
        self.stack.reserve(1024);

        self.output.tree.extra.clear();
        self.output.tree.nodes.clear();
        self.output.tree.nodes.reserve(source.len() / 8);
        self.output.errors.clear();

        self.advance_no_emit();
    }

    fn finish(&mut self) -> &mut Output {
        let root = self.stack.pop().expect("missing parse tree root on the stack");
        assert!(self.stack.is_empty(), "more than one root for parse tree");
        self.output.tree.nodes.push(root);
        &mut self.output
    }

    fn open(&mut self) -> OpenMark {
        OpenMark { stack_length: self.stack.len() }
    }

    fn close(&mut self, mark: OpenMark, tag: Tag) {
        debug_assert!(tag.is_syntax());

        let start = self.output.tree.nodes.len();
        self.output.tree.nodes.extend_from_slice(&self.stack[mark.stack_length..]);
        let end = self.output.tree.nodes.len();
        self.stack.truncate(mark.stack_length);

        self.stack.push(Node {
            tag,
            length: U24::from_usize(end - start).expect("syntax node has too many children"),
            offset: start.try_into().unwrap(),
        })
    }

    fn emit_error(&mut self, expected: Expected) {
        self.output.errors.push(Error {
            token_found: self.current_token,
            token_previous: self.previous_token,
            expected,
        })
    }

    fn advance_no_emit(&mut self) {
        self.fuel = Self::MAX_FUEL;
        self.previous_token = self.current_token;
        self.current_token = loop {
            let token = self.tokens.next();
            match token.tag {
                Tag::Comment | Tag::Preprocessor => self.output.tree.extra.push(token),
                _ => break token,
            }
        };
    }

    fn advance(&mut self) {
        self.stack.push(self.current_token);
        self.advance_no_emit();
    }

    fn advance_with_error(&mut self, mark: OpenMark, expected: Expected) {
        self.emit_error(expected);
        self.advance();
        self.close(mark, Tag::InvalidSyntax);

        match self.stack.as_slice() {
            [.., a, b] if a.tag == Tag::InvalidSyntax && b.tag == Tag::InvalidSyntax => {
                if a.children_range().end == b.children_range().start {
                    if let Some(combined_length) =
                        U24::from_usize(a.length.to_usize() + b.length.to_usize())
                    {
                        let combined = Node {
                            tag: Tag::InvalidSyntax,
                            offset: a.offset,
                            length: combined_length,
                        };
                        self.stack.truncate(self.stack.len() - 2);
                        self.stack.push(combined);
                    }
                }
            },
            _ => {},
        }
    }

    fn peek(&mut self) -> Tag {
        assert_ne!(self.fuel, 0, "parser ran out of fuel (got stuck in a loop)");
        self.fuel -= 1;
        self.current_token.tag
    }

    fn at(&mut self, token: Tag) -> bool {
        self.peek() == token
    }

    fn at_any(&mut self, set: TokenSet) -> bool {
        set.contains(self.peek())
    }

    fn consume(&mut self, token: Tag) -> bool {
        if self.at(token) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, token: Tag) {
        if !self.consume(token) {
            self.emit_error(Expected::Token(token));
        }
    }
}

#[derive(Clone, Copy)]
struct OpenMark {
    stack_length: usize,
}

const TOP_LEVEL_FIRST: TokenSet = TokenSet::new(&[
    Tag::KeywordDiagnostic,
    Tag::KeywordEnable,
    Tag::KeywordRequires,
    Tag::KeywordConstAssert,
    Tag::KeywordStruct,
    Tag::KeywordConst,
    Tag::KeywordOverride,
    Tag::KeywordVar,
    Tag::KeywordFn,
    Tag::SemiColon,
]);

pub fn parse_file<'parser, 'src>(
    parser: &'parser mut Parser<'src>,
    source: &'src str,
) -> &'parser mut Output {
    parser.begin(source);

    let m = parser.open();

    loop {
        match parser.peek() {
            Tag::Eof => break,

            Tag::KeywordDiagnostic => directive_diagnostic(parser),
            Tag::KeywordEnable => directive_enable(parser),
            Tag::KeywordRequires => directive_requires(parser),

            Tag::SemiColon => parser.advance(),

            Tag::KeywordAlias => {
                let m = parser.open();
                parser.advance();
                parser.expect(Tag::Identifier);
                parser.expect(Tag::Equal);
                type_specifier(parser);
                parser.expect(Tag::SemiColon);
                parser.close(m, Tag::DeclAlias);
            },

            Tag::KeywordConstAssert => {
                let m = parser.open();
                parser.advance();
                expression(parser);
                parser.expect(Tag::SemiColon);
                parser.close(m, Tag::DeclConstAssert);
            },

            Tag::KeywordStruct => {
                let m = parser.open();
                parser.advance();
                parser.expect(Tag::Identifier);

                {
                    let m = parser.open();
                    parser.expect(Tag::LCurly);
                    while parser.at(Tag::Identifier) || parser.at(Tag::AtSign) {
                        let m = parser.open();
                        attribute_list_maybe(parser, m);
                        parser.expect(Tag::Identifier);
                        parser.expect(Tag::Colon);
                        type_specifier(parser);
                        if !parser.at(Tag::RCurly) {
                            parser.expect(Tag::Comma);
                        }
                        parser.close(m, Tag::DeclStructField)
                    }
                    parser.expect(Tag::RCurly);
                    parser.close(m, Tag::DeclStructFieldList);
                }

                parser.close(m, Tag::DeclStruct);
            },

            Tag::KeywordConst => {
                let m = parser.open();
                variable_declaration(parser, m, Tag::KeywordConst);
            },

            _ => {
                let m = parser.open();
                attribute_list_maybe(parser, m);

                match parser.peek() {
                    tag @ (Tag::KeywordOverride | Tag::KeywordVar) => {
                        variable_declaration(parser, m, tag)
                    },

                    Tag::KeywordFn => {
                        parser.advance();
                        parser.expect(Tag::Identifier);

                        {
                            let m = parser.open();
                            parser.expect(Tag::LParen);
                            while parser.at(Tag::Identifier) || parser.at(Tag::AtSign) {
                                let m = parser.open();
                                attribute_list_maybe(parser, m);
                                parser.expect(Tag::Identifier);
                                parser.expect(Tag::Colon);
                                type_specifier(parser);
                                if !parser.at(Tag::RParen) {
                                    parser.expect(Tag::Comma);
                                }
                                parser.close(m, Tag::DeclFnParameter);
                            }
                            parser.expect(Tag::RParen);
                            parser.close(m, Tag::DeclFnParameterList);
                        }

                        let mark_output = parser.open();
                        if parser.consume(Tag::ThinArrowRight) {
                            attribute_list_maybe(parser, mark_output);
                            type_specifier(parser);
                            parser.close(mark_output, Tag::DeclFnOutput);
                        }

                        statement_block(parser);

                        parser.close(m, Tag::DeclFn);
                    },

                    _ => parser.advance_with_error(m, Expected::Declaration),
                }
            },
        }
    }

    parser.close(m, Tag::Root);

    parser.finish()
}

pub(crate) fn parse_type_specifier<'a, 'src>(
    parser: &'a mut Parser<'src>,
    source: &'src str,
) -> &'a mut Output {
    parser.begin(source);
    type_specifier(parser);
    parser.finish()
}

const STATEMENT_FIRST: TokenSet = EXPRESSION_FIRST.with_many(&[
    Tag::KeywordReturn,
    Tag::KeywordConstAssert,
    Tag::KeywordDiscard,
    Tag::KeywordBreak,
    Tag::KeywordContinue,
    Tag::KeywordContinuing,
    Tag::KeywordLet,
    Tag::KeywordVar,
    Tag::KeywordConst,
    Tag::KeywordFn,
    Tag::KeywordIf,
    Tag::KeywordFor,
    Tag::KeywordLoop,
    Tag::KeywordWhile,
    Tag::KeywordSwitch,
    Tag::SemiColon,
    Tag::LCurly,
]);

fn statement(parser: &mut Parser) {
    match parser.peek() {
        Tag::KeywordReturn => {
            let m = parser.open();
            parser.advance();
            expression(parser);
            parser.expect(Tag::SemiColon);
            parser.close(m, Tag::StmtReturn);
        },

        Tag::KeywordConstAssert => {
            let m = parser.open();
            parser.advance();
            expression(parser);
            parser.expect(Tag::SemiColon);
            parser.close(m, Tag::DeclConstAssert);
        },

        Tag::KeywordDiscard => {
            let m = parser.open();
            parser.advance();
            parser.expect(Tag::SemiColon);
            parser.close(m, Tag::StmtDiscard);
        },

        Tag::KeywordBreak => {
            let m = parser.open();
            parser.advance();
            if parser.consume(Tag::KeywordIf) {
                expression(parser);
            }
            parser.expect(Tag::SemiColon);
            parser.close(m, Tag::StmtBreak);
        },
        Tag::KeywordContinue => {
            let m = parser.open();
            parser.advance();
            parser.expect(Tag::SemiColon);
            parser.close(m, Tag::StmtContinue);
        },
        Tag::KeywordContinuing => {
            let m = parser.open();
            parser.advance();
            statement(parser);
            parser.close(m, Tag::StmtContinuing);
        },

        tag @ (Tag::KeywordLet | Tag::KeywordConst | Tag::KeywordVar) => {
            let m = parser.open();
            variable_declaration(parser, m, tag)
        },

        Tag::SemiColon => parser.advance(),

        _ if parser.at_any(EXPRESSION_FIRST) => statement_expression(parser, true),

        _ => {
            let m = parser.open();
            attribute_list_maybe(parser, m);

            match parser.peek() {
                Tag::LCurly => statement_block_post_attributes(parser, m),

                Tag::KeywordFor => {
                    parser.advance();

                    {
                        parser.expect(Tag::LParen);
                        match parser.peek() {
                            tag @ (Tag::KeywordLet | Tag::KeywordConst | Tag::KeywordVar) => {
                                let m = parser.open();
                                variable_declaration(parser, m, tag)
                            },
                            _ if parser.at_any(EXPRESSION_FIRST) => {
                                statement_expression(parser, true)
                            },
                            Tag::SemiColon => parser.advance(),
                            _ => parser.emit_error(Expected::Statement),
                        }
                        expression(parser);
                        parser.expect(Tag::SemiColon);
                        statement_expression(parser, false);
                        parser.expect(Tag::RParen);
                    }

                    statement_block(parser);

                    parser.close(m, Tag::StmtFor);
                },

                Tag::KeywordIf => {
                    let mut mark_branch = parser.open();
                    loop {
                        let has_condition = parser.consume(Tag::KeywordIf);
                        if has_condition {
                            expression(parser);
                        }

                        statement_block(parser);

                        parser.close(mark_branch, Tag::StmtIfBranch);
                        mark_branch = parser.open();

                        if !(has_condition && parser.consume(Tag::KeywordElse)) {
                            break;
                        }
                    }
                    parser.close(m, Tag::StmtIf);
                },

                Tag::KeywordLoop => {
                    parser.advance();
                    statement_block(parser);
                    parser.close(m, Tag::StmtLoop);
                },

                Tag::KeywordWhile => {
                    parser.advance();
                    expression(parser);
                    statement_block(parser);
                    parser.close(m, Tag::StmtWhile);
                },

                Tag::KeywordSwitch => {
                    parser.advance();
                    expression(parser);

                    let mark_body = parser.open();
                    parser.expect(Tag::LCurly);
                    while parser.at(Tag::KeywordCase) || parser.at(Tag::KeywordDefault) {
                        let m = parser.open();

                        const CASE_FOLLOWS: TokenSet =
                            TokenSet::new(&[Tag::Colon, Tag::LCurly, Tag::AtSign]);

                        if parser.consume(Tag::KeywordCase) {
                            while parser
                                .at_any(const { EXPRESSION_FIRST.with(Tag::KeywordDefault) })
                            {
                                let m = parser.open();
                                if !parser.consume(Tag::KeywordDefault) {
                                    expression(parser);
                                }
                                if !parser.at_any(CASE_FOLLOWS) {
                                    parser.expect(Tag::Comma);
                                }
                                parser.close(m, Tag::StmtSwitchBranchCase);
                            }
                        } else {
                            parser.expect(Tag::KeywordDefault);
                        }

                        parser.consume(Tag::Colon);
                        statement_block(parser);

                        parser.close(m, Tag::StmtSwitchBranch);
                    }
                    parser.expect(Tag::RCurly);
                    parser.close(mark_body, Tag::StmtSwitchBranchList);

                    parser.close(m, Tag::StmtSwitch);
                },

                _ => parser.advance_with_error(m, Expected::Statement),
            }
        },
    }
}

fn variable_declaration(parser: &mut Parser, mark: OpenMark, tag: Tag) {
    parser.expect(tag);

    if tag == Tag::KeywordVar && parser.at(Tag::TemplateListStart) {
        template_list(parser);
    }
    parser.expect(Tag::Identifier);
    if parser.consume(Tag::Colon) {
        type_specifier(parser);
    }

    if tag == Tag::KeywordVar {
        if parser.consume(Tag::Equal) {
            expression(parser);
        }
    } else {
        parser.expect(Tag::Equal);
        expression(parser);
    }

    parser.expect(Tag::SemiColon);

    parser.close(
        mark,
        match tag {
            Tag::KeywordVar => Tag::DeclVar,
            Tag::KeywordLet => Tag::StmtLet,
            Tag::KeywordConst => Tag::DeclConst,
            Tag::KeywordOverride => Tag::DeclOverride,
            _ => unreachable!(),
        },
    )
}

pub const ASSIGNMENT_OPS: TokenSet = TokenSet::new(&[
    Tag::Equal,
    Tag::LessLessEqual,
    Tag::GreaterGreaterEqual,
    Tag::PlusEqual,
    Tag::MinusEqual,
    Tag::AsteriskEqual,
    Tag::SlashEqual,
    Tag::PercentEqual,
    Tag::ChevronEqual,
    Tag::AmpersandEqual,
    Tag::BarEqual,
]);

fn statement_expression(parser: &mut Parser, semi: bool) {
    let m = parser.open();
    expression(parser);

    if parser.at_any(ASSIGNMENT_OPS) {
        parser.advance();
        expression(parser);
        if semi {
            parser.expect(Tag::SemiColon);
        }
        parser.close(m, Tag::StmtAssign);
    } else if parser.at(Tag::PlusPlus) {
        parser.advance();
        if semi {
            parser.expect(Tag::SemiColon);
        }
        parser.close(m, Tag::StmtIncrement);
    } else if parser.at(Tag::MinusMinus) {
        parser.advance();
        if semi {
            parser.expect(Tag::SemiColon);
        }
        parser.close(m, Tag::StmtDecrement);
    } else {
        if semi {
            parser.expect(Tag::SemiColon);
        }
        parser.close(m, Tag::StmtExpr);
    }
}

fn statement_block(parser: &mut Parser) {
    let m = parser.open();
    attribute_list_maybe(parser, m);
    statement_block_post_attributes(parser, m);
}

fn statement_block_post_attributes(parser: &mut Parser, mark: OpenMark) {
    parser.expect(Tag::LCurly);
    while !parser.at(Tag::Eof) && !parser.at(Tag::RCurly) {
        if parser.at_any(STATEMENT_FIRST) {
            statement(parser);
        } else if parser.at_any(TOP_LEVEL_FIRST) {
            break;
        } else {
            let m = parser.open();
            parser.advance_with_error(m, Expected::Statement);
        }
    }
    parser.expect(Tag::RCurly);
    parser.close(mark, Tag::StmtBlock);
}

fn attribute_list_maybe(parser: &mut Parser, mark: OpenMark) {
    let mut has_attribute = false;
    while parser.at(Tag::AtSign) {
        attribute(parser);
        has_attribute = true;
    }
    if has_attribute {
        parser.close(mark, Tag::AttributeList);
    }
}

fn attribute(parser: &mut Parser) {
    let m = parser.open();
    parser.expect(Tag::AtSign);
    parser.expect(Tag::Identifier);
    if parser.at(Tag::LParen) {
        argument_list(parser);
    }
    parser.close(m, Tag::Attribute);
}

fn argument_list(parser: &mut Parser) {
    let m = parser.open();
    parser.expect(Tag::LParen);
    loop {
        let m = parser.open();
        if !expression_maybe(parser) {
            break;
        }
        if !parser.at(Tag::RParen) {
            parser.expect(Tag::Comma);
        }
        parser.close(m, Tag::Argument);
    }
    parser.expect(Tag::RParen);
    parser.close(m, Tag::ArgumentList);
}

const EXPRESSION_FIRST: TokenSet = EXPRESSION_PRIMARY_FIRST.union(EXPRESSION_UNARY_OPS);

fn expression(parser: &mut Parser) {
    if !expression_maybe(parser) {
        parser.emit_error(Expected::Expression);
    }
}

fn expression_maybe(parser: &mut Parser) -> bool {
    if parser.at_any(EXPRESSION_FIRST) {
        expression_infix(parser, 0);
        true
    } else {
        false
    }
}

pub const EXPRESSION_INFIX_OPS: TokenSet = TokenSet::new(&[
    Tag::Asterisk,
    Tag::Slash,
    Tag::Percent,
    Tag::LessLess,
    Tag::GreaterGreater,
    Tag::Chevron,
    Tag::Ampersand,
    Tag::Bar,
    Tag::Plus,
    Tag::Minus,
    Tag::Less,
    Tag::LessEqual,
    Tag::Greater,
    Tag::GreaterEqual,
    Tag::EqualEqual,
    Tag::ExclamationEqual,
    Tag::AmpersandAmpersand,
    Tag::BarBar,
]);

fn expression_infix(parser: &mut Parser, left_binding_power: u8) {
    // higher binding power means it binds tighter. Multiplication binds tighter than addition.
    fn binding_power(op: Tag) -> u8 {
        match op {
            Tag::Asterisk | Tag::Slash | Tag::Percent => 7,

            Tag::LessLess | Tag::GreaterGreater => 6,

            Tag::Chevron | Tag::Ampersand | Tag::Bar => 5,

            Tag::Plus | Tag::Minus => 4,

            Tag::Less
            | Tag::LessEqual
            | Tag::Greater
            | Tag::GreaterEqual
            | Tag::EqualEqual
            | Tag::ExclamationEqual => 3,

            Tag::AmpersandAmpersand => 2,

            Tag::BarBar => 1,

            _ => 0,
        }
    }

    let lhs = parser.open();

    expression_prefix(parser);

    loop {
        let right_binding_power = binding_power(parser.peek());
        if left_binding_power >= right_binding_power {
            break;
        }
        parser.advance();
        expression_infix(parser, right_binding_power);
        parser.close(lhs, Tag::ExprInfix);
    }
}

const EXPRESSION_UNARY_OPS: TokenSet =
    TokenSet::new(&[Tag::Minus, Tag::Ampersand, Tag::Asterisk, Tag::Exclamation, Tag::Tilde]);

fn expression_prefix(parser: &mut Parser) {
    let m = parser.open();

    let mut has_unary = false;
    while parser.at_any(EXPRESSION_UNARY_OPS) {
        parser.advance();
        has_unary = true;
    }

    expression_suffix(parser);

    if has_unary {
        parser.close(m, Tag::ExprPrefix);
    }
}

fn expression_suffix(parser: &mut Parser) {
    let lhs = parser.open();
    expression_primary(parser);

    loop {
        match parser.peek() {
            Tag::LParen => {
                argument_list(parser);
                parser.close(lhs, Tag::ExprCall);
            },

            Tag::Dot => {
                parser.advance();
                parser.expect(Tag::Identifier);
                parser.close(lhs, Tag::ExprMember);
            },

            Tag::LBracket => {
                parser.advance();
                expression(parser);
                parser.expect(Tag::RBracket);
                parser.close(lhs, Tag::ExprIndex);
            },

            _ => break,
        }
    }
}

const EXPRESSION_PRIMARY_FIRST: TokenSet = TokenSet::new(&[
    Tag::Identifier,
    Tag::IntegerDecimal,
    Tag::IntegerHex,
    Tag::FloatDecimal,
    Tag::FloatHex,
    Tag::LParen,
    Tag::KeywordTrue,
    Tag::KeywordFalse,
]);

fn expression_primary(parser: &mut Parser) {
    match parser.peek() {
        Tag::Identifier => identifier_maybe_with_template(parser),

        Tag::IntegerDecimal
        | Tag::IntegerHex
        | Tag::FloatDecimal
        | Tag::FloatHex
        | Tag::KeywordTrue
        | Tag::KeywordFalse => parser.advance(),

        Tag::LParen => {
            let m = parser.open();
            parser.expect(Tag::LParen);
            expression(parser);
            parser.expect(Tag::RParen);
            parser.close(m, Tag::ExprParens);
        },

        token => {
            assert!(!EXPRESSION_PRIMARY_FIRST.contains(token));
            parser.emit_error(Expected::Expression)
        },
    }
}

fn type_specifier(parser: &mut Parser) {
    identifier_maybe_with_template(parser)
}

fn identifier_maybe_with_template(parser: &mut Parser) {
    let m = parser.open();
    parser.expect(Tag::Identifier);
    if parser.at(Tag::TemplateListStart) {
        template_list(parser);
        parser.close(m, Tag::IdentifierWithTemplate);
    }
}

fn template_list(parser: &mut Parser) {
    let m = parser.open();
    parser.expect(Tag::TemplateListStart);
    loop {
        let m = parser.open();
        if !expression_maybe(parser) {
            break;
        }
        if !parser.at(Tag::TemplateListEnd) {
            parser.expect(Tag::Comma);
        }
        parser.close(m, Tag::TemplateParameter);
    }
    parser.expect(Tag::TemplateListEnd);
    parser.close(m, Tag::TemplateList);
}

fn directive_diagnostic(parser: &mut Parser) {
    let m = parser.open();
    assert!(parser.consume(Tag::KeywordDiagnostic));
    parser.expect(Tag::LParen);
    while parser.at(Tag::Identifier) {
        let m = parser.open();
        parser.expect(Tag::Identifier);
        if parser.consume(Tag::Dot) {
            parser.expect(Tag::Identifier);
        }
        if !parser.at(Tag::RParen) {
            parser.expect(Tag::Comma);
        }
        parser.close(m, Tag::DirectiveDiagnosticName)
    }
    parser.expect(Tag::RParen);
    parser.expect(Tag::SemiColon);
    parser.close(m, Tag::DirectiveDiagnostic);
}

fn directive_enable(parser: &mut Parser) {
    let m = parser.open();
    parser.advance();
    while parser.at(Tag::Identifier) {
        let m = parser.open();
        parser.expect(Tag::Identifier);
        if !parser.at(Tag::SemiColon) {
            parser.expect(Tag::Comma);
        }
        parser.close(m, Tag::DirectiveEnableName)
    }
    parser.expect(Tag::SemiColon);
    parser.close(m, Tag::DirectiveEnable);
}

fn directive_requires(parser: &mut Parser) {
    let m = parser.open();
    parser.advance();
    while parser.at(Tag::Identifier) {
        let m = parser.open();
        parser.expect(Tag::Identifier);
        if !parser.at(Tag::SemiColon) {
            parser.expect(Tag::Comma);
        }
        parser.close(m, Tag::DirectiveRequiresName)
    }
    parser.expect(Tag::SemiColon);
    parser.close(m, Tag::DirectiveRequires);
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::*;

    fn check(source: &str, expected: expect_test::Expect) {
        use std::fmt::Write;

        let mut parser = Parser::new();
        let output = parse_file(&mut parser, source);

        let mut text = String::new();

        if !output.errors.is_empty() {
            writeln!(text, "===== ERRORS =====").unwrap();
            for error in output.errors.iter() {
                let before = &source[..error.token_found.offset as usize];
                let line = before.lines().count();
                let line_start = before.rfind('\n').map(|x| x + 1).unwrap_or(0);
                let column = before[line_start..].chars().count();
                writeln!(text, "[{line}:{column}] {}", error.message(source)).unwrap();
            }

            writeln!(text, "===== SYNTAX =====").unwrap();
        }

        output
            .tree
            .traverse_pre_order(|_, node, depth| {
                for _ in 0..depth {
                    text.push_str("  ");
                }
                if node.tag.is_syntax() {
                    writeln!(text, "{:?}", node.tag)
                } else {
                    writeln!(text, "{:?} {:?}", node.tag, &source[node.byte_range()])
                }
            })
            .unwrap();

        expected.assert_eq(&text);
    }

    #[test]
    fn decl_fn() {
        check(
            indoc::indoc! {r#"
                fn main() {}
            "#},
            expect![[r#"
                Root
                  DeclFn
                    KeywordFn "fn"
                    Identifier "main"
                    DeclFnParameterList
                      LParen "("
                      RParen ")"
                    StmtBlock
                      LCurly "{"
                      RCurly "}"
            "#]],
        );
    }

    #[test]
    fn decl_fn_missing_parens() {
        check(
            indoc::indoc! {r#"
                fn main( {}
            "#},
            expect![[r#"
                ===== ERRORS =====
                [1:9] expected `)`, but found `{`
                ===== SYNTAX =====
                Root
                  DeclFn
                    KeywordFn "fn"
                    Identifier "main"
                    DeclFnParameterList
                      LParen "("
                    StmtBlock
                      LCurly "{"
                      RCurly "}"
            "#]],
        );
    }

    #[test]
    fn block_in_arguments() {
        check(
            indoc::indoc! {r#"
                fn main() {
                    cross(123{);
                }
            "#},
            expect![[r#"
                ===== ERRORS =====
                [2:13] expected `,`, but found `{`
                [2:13] expected `)`, but found `{`
                [2:13] expected `;`, but found `{`
                [2:14] expected a statement, but found `)`
                [3:0] expected `}`, but found the end of file
                ===== SYNTAX =====
                Root
                  DeclFn
                    KeywordFn "fn"
                    Identifier "main"
                    DeclFnParameterList
                      LParen "("
                      RParen ")"
                    StmtBlock
                      LCurly "{"
                      StmtExpr
                        ExprCall
                          Identifier "cross"
                          ArgumentList
                            LParen "("
                            Argument
                              IntegerDecimal "123"
                      StmtBlock
                        LCurly "{"
                        InvalidSyntax
                          RParen ")"
                        SemiColon ";"
                        RCurly "}"
            "#]],
        );
    }

    #[test]
    fn for_loop() {
        check(
            indoc::indoc! {r#"
                fn foo() -> i32 {
                    for (var i = 0i; i < 10; i += 1) {
                        if i == 8 {
                            return i;
                        }
                    }

                    var i = 0i;
                    for (; i < 10; i++) {}
                }
            "#},
            expect![[r#"
                Root
                  DeclFn
                    KeywordFn "fn"
                    Identifier "foo"
                    DeclFnParameterList
                      LParen "("
                      RParen ")"
                    DeclFnOutput
                      ThinArrowRight "->"
                      Identifier "i32"
                    StmtBlock
                      LCurly "{"
                      StmtFor
                        KeywordFor "for"
                        LParen "("
                        DeclVar
                          KeywordVar "var"
                          Identifier "i"
                          Equal "="
                          IntegerDecimal "0i"
                          SemiColon ";"
                        ExprInfix
                          Identifier "i"
                          Less "<"
                          IntegerDecimal "10"
                        SemiColon ";"
                        StmtAssign
                          Identifier "i"
                          PlusEqual "+="
                          IntegerDecimal "1"
                        RParen ")"
                        StmtBlock
                          LCurly "{"
                          StmtIf
                            StmtIfBranch
                              KeywordIf "if"
                              ExprInfix
                                Identifier "i"
                                EqualEqual "=="
                                IntegerDecimal "8"
                              StmtBlock
                                LCurly "{"
                                StmtReturn
                                  KeywordReturn "return"
                                  Identifier "i"
                                  SemiColon ";"
                                RCurly "}"
                          RCurly "}"
                      DeclVar
                        KeywordVar "var"
                        Identifier "i"
                        Equal "="
                        IntegerDecimal "0i"
                        SemiColon ";"
                      StmtFor
                        KeywordFor "for"
                        LParen "("
                        SemiColon ";"
                        ExprInfix
                          Identifier "i"
                          Less "<"
                          IntegerDecimal "10"
                        SemiColon ";"
                        StmtIncrement
                          Identifier "i"
                          PlusPlus "++"
                        RParen ")"
                        StmtBlock
                          LCurly "{"
                          RCurly "}"
                      RCurly "}"
            "#]],
        )
    }
}
