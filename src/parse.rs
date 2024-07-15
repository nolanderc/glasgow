mod token;

use std::num::NonZeroUsize;

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

    DirectiveDiagnostic,
    DirectiveDiagnosticName,

    DirectiveEnable,
    DirectiveEnableName,

    DirectiveRequires,
    DirectiveRequiresName,

    ArgumentList,
    Argument,
    Attribute,

    ExprParens,
    ExprPrefix,

    ExprCall,
    AttributeList,
    TemplateList,
    TemplateParameter,
    IdentifierWithTemplate,
    ExprInfix,
    ExprMember,
    ExprIndex,
    DeclAlias,
    ConstAssert,

    DeclStruct,
    DeclStructFieldList,
    DeclStructField,
    DeclConst,
    DeclVar,

    DeclFn,
    DeclFnParameterList,
    DeclFnParameter,
    DeclFnOutput,
    StmtBlock,
    StmtReturn,
    StmtDiscard,
    StmtBreak,
    StmtContinue,
    StmtExpr,
    StmtLet,
    StmtContinuing,
    StmtAssign,
    StmtFor,
    StmtIf,
    StmtIfBranch,
    StmtLoop,
    StmtWhile,
    StmtSwitch,
    StmtSwitchBranch,
    StmtSwitchCaseSelector,
    // ^^^^^ syntax ^^^^^ //
}

impl Tag {
    pub const fn is_token(self) -> bool {
        (self as u8) < (Tag::InvalidSyntax as u8)
    }

    pub const fn is_syntax(self) -> bool {
        !self.is_token()
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Node {
    tag: Tag,
    length: Length,
    offset: u32,
}

impl Node {
    pub fn tag(self) -> Tag {
        self.tag
    }

    pub fn byte_range(self, source: &str) -> std::ops::Range<usize> {
        debug_assert!(self.tag.is_token());

        let offset = self.offset as usize;
        let length = self
            .length
            .get_token()
            .or_else(|| token::variable_length(&source[offset..]))
            .map(|x| x.get())
            .or_else(|| if self.tag == Tag::Eof { Some(0) } else { None })
            .expect("could not get length of token");

        offset..offset + length
    }

    pub fn slice(self, source: &str) -> &str {
        let range = self.byte_range(source);
        &source[range]
    }

    pub fn children(self) -> std::ops::Range<usize> {
        debug_assert!(self.tag.is_syntax());
        let offset = self.offset as usize;
        offset..offset + self.length.get_syntax() as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Length(U24);

impl Length {
    fn new_token(length: usize) -> Length {
        Length(U24::from_usize(length).unwrap_or(U24::ZERO))
    }

    pub fn get_token(self) -> Option<NonZeroUsize> {
        NonZeroUsize::new(self.0.to_usize())
    }

    fn new_syntax(length: usize) -> Option<Length> {
        Some(Length(U24::from_usize(length)?))
    }

    pub fn get_syntax(self) -> u32 {
        self.0.to_u32()
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

pub type NodeIndex = u32;
pub type Depth = usize;

impl Tree {
    pub fn node(&self, index: NodeIndex) -> Node {
        self.nodes[index as usize]
    }

    #[cfg(test)]
    pub fn traverse_pre_order<E>(
        &self,
        mut visit: impl FnMut(NodeIndex, Node, Depth) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut stack = Vec::with_capacity(32);

        stack.push(self.nodes.len() - 1..self.nodes.len());

        while let Some(top) = stack.last_mut() {
            let Some(curr) = top.next() else {
                stack.pop();
                continue;
            };

            let depth = stack.len() - 1;
            let node = self.nodes[curr];
            visit(curr as NodeIndex, node, depth)?;

            if node.tag.is_syntax() {
                stack.push(node.children());
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Error {
    pub token: Node,
    pub expected: Expected,
}

#[derive(Debug, Clone, Copy)]
pub enum Expected {
    Token(Tag),
    Declaration,
    Expression,
    Statement,
}

pub struct Parser<'src> {
    tokens: token::Tokenizer<'src>,
    current_token: Node,

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
    const MAX_FUEL: u32 = 16;

    pub fn new() -> Parser<'static> {
        Parser {
            tokens: token::Tokenizer::empty(),
            current_token: Node { tag: Tag::Eof, length: Length(U24::ZERO), offset: 0 },
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
            length: Length::new_syntax(end - start).unwrap_or(Length(U24::ZERO)),
            offset: start.try_into().unwrap(),
        })
    }

    fn emit_error(&mut self, expected: Expected) {
        self.output.errors.push(Error { token: self.current_token, expected })
    }

    fn advance_no_emit(&mut self) {
        self.fuel = Self::MAX_FUEL;
        self.current_token = loop {
            let token = self.tokens.next();
            match token.tag {
                Tag::Comment => self.output.tree.extra.push(token),
                _ => break token,
            }
        };
    }

    fn advance(&mut self) {
        self.stack.push(self.current_token);
        self.advance_no_emit();
    }

    fn advance_with_error(&mut self, expected: Expected) {
        let m = self.open();
        self.emit_error(expected);
        self.advance();
        self.close(m, Tag::InvalidSyntax);
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

    fn eof(&mut self) -> bool {
        self.at(Tag::Eof)
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
                parser.close(m, Tag::ConstAssert);
            },

            Tag::KeywordStruct => {
                let m = parser.open();
                parser.advance();
                parser.expect(Tag::Identifier);

                {
                    let m = parser.open();
                    parser.expect(Tag::LCurly);
                    while !parser.eof() && !parser.at(Tag::RCurly) {
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
                parser.advance();
                parser.expect(Tag::Identifier);
                if parser.consume(Tag::Colon) {
                    type_specifier(parser);
                }
                parser.expect(Tag::Equal);
                expression(parser);
                parser.expect(Tag::SemiColon);
                parser.close(m, Tag::DeclConst);
            },

            _ => {
                let m = parser.open();
                attribute_list_maybe(parser, m);

                match parser.peek() {
                    Tag::KeywordFn => {
                        parser.advance();
                        parser.expect(Tag::Identifier);

                        {
                            let m = parser.open();
                            parser.expect(Tag::LParen);
                            while !parser.eof() && !parser.at(Tag::RParen) {
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

                        if parser.consume(Tag::ThinArrowRight) {
                            let m = parser.open();
                            attribute_list_maybe(parser, m);
                            type_specifier(parser);
                            parser.close(m, Tag::DeclFnOutput);
                        }

                        statement_block(parser);

                        parser.close(m, Tag::DeclFn);
                    },

                    _ => parser.advance_with_error(Expected::Declaration),
                }
            },
        }
    }

    parser.close(m, Tag::Root);

    parser.finish()
}

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
            parser.close(m, Tag::ConstAssert);
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
            statement_variable_declaration(parser, tag)
        },

        Tag::SemiColon => parser.advance(),

        _ if parser.at_any(EXPRESSION_FIRST) => statement_expression(parser),

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
                                statement_variable_declaration(parser, tag)
                            },
                            _ if parser.at_any(EXPRESSION_FIRST) => statement_expression(parser),
                            _ => parser.emit_error(Expected::Statement),
                        }
                        expression(parser);
                        parser.expect(Tag::SemiColon);
                        statement_expression(parser);
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

                    parser.expect(Tag::LCurly);
                    while !parser.eof() && !parser.at(Tag::RCurly) {
                        let m = parser.open();

                        const CASE_FOLLOWS: TokenSet =
                            TokenSet::new(&[Tag::Colon, Tag::LCurly, Tag::AtSign]);

                        if parser.consume(Tag::KeywordCase) {
                            while !parser.eof() && !parser.at_any(CASE_FOLLOWS) {
                                let m = parser.open();
                                if !parser.consume(Tag::KeywordDefault) {
                                    expression(parser);
                                }
                                if !parser.at_any(CASE_FOLLOWS) {
                                    parser.expect(Tag::Comma);
                                }
                                parser.close(m, Tag::StmtSwitchCaseSelector);
                            }
                        } else {
                            parser.expect(Tag::KeywordDefault);
                        }

                        parser.consume(Tag::Colon);
                        statement_block(parser);

                        parser.close(m, Tag::StmtSwitchBranch);
                    }
                    parser.expect(Tag::RCurly);

                    parser.close(m, Tag::StmtSwitch);
                },

                _ => parser.advance_with_error(Expected::Statement),
            }
        },
    }
}

fn statement_variable_declaration(parser: &mut Parser, tag: Tag) {
    let m = parser.open();

    parser.advance();
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
        m,
        match tag {
            Tag::KeywordVar => Tag::DeclVar,
            Tag::KeywordLet => Tag::StmtLet,
            Tag::KeywordConst => Tag::DeclConst,
            _ => unreachable!(),
        },
    )
}

const ASSIGNMENT_OP: TokenSet = TokenSet::new(&[
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

const INCREMENT_DECREMENT_OP: TokenSet = TokenSet::new(&[Tag::PlusPlus, Tag::MinusMinus]);

fn statement_expression(parser: &mut Parser) {
    let m = parser.open();
    expression(parser);

    if parser.at_any(ASSIGNMENT_OP) {
        parser.advance();
        expression(parser);
        parser.expect(Tag::SemiColon);
        parser.close(m, Tag::StmtAssign);
    } else if parser.at_any(INCREMENT_DECREMENT_OP) {
        parser.advance();
        parser.expect(Tag::SemiColon);
        parser.close(m, Tag::StmtExpr);
    } else {
        parser.expect(Tag::SemiColon);
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
    while !parser.eof() && !parser.at(Tag::RCurly) {
        statement(parser);
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
    while !parser.eof() && !parser.at(Tag::RParen) {
        let m = parser.open();
        expression(parser);
        if !parser.at(Tag::RParen) {
            parser.expect(Tag::Comma);
        }
        parser.close(m, Tag::Argument);
    }
    parser.expect(Tag::RParen);
    parser.close(m, Tag::ArgumentList);
}

const EXPRESSION_FIRST: TokenSet = EXPRESSION_PRIMARY_FIRST.union(EXPRESSION_UNARY_OP);

fn expression(parser: &mut Parser) {
    if !parser.at_any(EXPRESSION_PRIMARY_FIRST) {
        parser.emit_error(Expected::Expression);
        return;
    }

    expression_infix(parser, 0);
}

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

const EXPRESSION_UNARY_OP: TokenSet =
    TokenSet::new(&[Tag::Minus, Tag::Ampersand, Tag::Asterisk, Tag::Exclamation, Tag::Tilde]);

fn expression_prefix(parser: &mut Parser) {
    let m = parser.open();

    let mut has_unary = false;
    while parser.at_any(EXPRESSION_UNARY_OP) {
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
    while !parser.eof() && !parser.at(Tag::TemplateListEnd) {
        let m = parser.open();
        expression(parser);
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
    while !parser.eof() && !parser.at(Tag::RParen) {
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
    while !parser.eof() && !parser.at(Tag::SemiColon) {
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
    while !parser.eof() && !parser.at(Tag::SemiColon) {
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
                writeln!(text, "{error:?}").unwrap();
            }
        }

        writeln!(text, "===== SYNTAX =====").unwrap();
        output
            .tree
            .traverse_pre_order(|_, node, depth| {
                for _ in 0..depth {
                    text.push_str("  ");
                }
                if node.tag.is_syntax() {
                    writeln!(text, "{:?}", node.tag)
                } else {
                    writeln!(text, "{:?} {:?}", node.tag, node.slice(source))
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
                      RCurly "}"
            "#]],
        );
    }
}
