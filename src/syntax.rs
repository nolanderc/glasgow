use crate::parse::{self, Tree};

#[derive(Debug, Clone, Copy)]
pub struct SyntaxNode {
    parse: parse::Node,
    index: parse::NodeIndex,
}

impl SyntaxNode {
    pub fn new(parse: parse::Node, index: parse::NodeIndex) -> Self {
        Self { parse, index }
    }

    pub fn parse_node(self) -> parse::Node {
        self.parse
    }

    pub fn index(self) -> parse::NodeIndex {
        self.index
    }

    pub fn children(
        self,
        tree: &Tree,
    ) -> impl ExactSizeIterator<Item = SyntaxNode> + DoubleEndedIterator + '_ {
        tree.children(self.parse)
            .iter()
            .zip(self.parse.children())
            .map(|(node, index)| SyntaxNode { parse: *node, index })
    }
}

pub trait SyntaxNodeMatch: Sized {
    fn new(node: SyntaxNode) -> Option<Self>;

    fn node(self) -> SyntaxNode;

    fn parse_node(self) -> parse::Node {
        self.node().parse_node()
    }

    fn index(self) -> parse::NodeIndex {
        self.node().index()
    }

    fn from_tree(tree: &Tree, index: parse::NodeIndex) -> Option<Self> {
        Self::new(SyntaxNode { parse: tree.node(index), index })
    }
}

pub trait Extract {
    type Data;
    fn extract(self, tree: &Tree) -> Self::Data;
}

macro_rules! syntax_node_simple {
    ( $name:ident ) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name(SyntaxNode);

        impl SyntaxNodeMatch for $name {
            #[inline(always)]
            fn new(node: SyntaxNode) -> Option<Self> {
                if node.parse.tag() == parse::Tag::$name {
                    Some(Self(node))
                } else {
                    None
                }
            }

            fn node(self) -> SyntaxNode {
                self.0
            }
        }
    };

    ( $name:ident, struct $extracted:ident { $( $field:ident : $fieldty:ty ),* $(,)? } ) => {
        syntax_node_simple!($name);

        #[allow(dead_code)]
        #[derive(Debug)]
        pub struct $extracted {
            $( pub $field: Option<$fieldty> ),*
        }

        #[allow(dead_code)]
        impl Extract for $name {
            type Data = $extracted;
            fn extract(self, tree: &Tree) -> $extracted {
                let mut children = self.0.children(tree).peekable();
                $( let $field = try_extract::<$fieldty>(&mut children); )*
                $extracted { $( $field ),* }
            }
        }
    };
}

#[inline(always)]
pub fn try_extract<T: SyntaxNodeMatch>(
    nodes: &mut std::iter::Peekable<impl Iterator<Item = SyntaxNode>>,
) -> Option<T> {
    loop {
        let node = *nodes.peek()?;
        if matches!(node.parse.tag(), parse::Tag::InvalidToken | parse::Tag::InvalidSyntax) {
            nodes.next();
            continue;
        }

        let x = T::new(node)?;
        nodes.next();
        return Some(x);
    }
}

macro_rules! syntax_node_enum {
    ( enum $name:ident { $( $variant:ident ($inner:ty)  ),* $(,)? } ) => {
        #[allow(dead_code)]
        #[derive(Debug, Clone, Copy)]
        pub enum $name {
            $( $variant($inner), )*
        }

        impl SyntaxNodeMatch for $name {
            #[inline]
            fn new(node: SyntaxNode) -> Option<Self> {
                $(
                    if let Some(x) = <$inner as SyntaxNodeMatch>::new(node) {
                        return Some(Self::$variant(x));
                    }
                )*

                None
            }

            fn node(self) -> SyntaxNode {
                match self {
                    $( Self::$variant(inner) => inner.node() ),*
                }
            }
        }
    };
}

#[derive(Debug, Clone, Copy)]
pub struct TokenNode<const TAG: u8>(SyntaxNode);

impl<const TAG: u8> SyntaxNodeMatch for TokenNode<TAG> {
    fn new(node: SyntaxNode) -> Option<Self> {
        if node.parse.tag() as u8 == TAG {
            Some(TokenNode(node))
        } else {
            None
        }
    }

    fn node(self) -> SyntaxNode {
        self.0
    }
}

impl<const TAG: u8> TokenNode<TAG> {
    pub fn byte_range(self) -> std::ops::Range<usize> {
        self.0.parse.byte_range()
    }
}

macro_rules! Token {
    ($tag:ident) => {
        $crate::syntax::TokenNode<{ parse::Tag::$tag as u8 }>
    };
}

pub(crate) use Token;

pub fn root(tree: &Tree) -> Root {
    Root::new(SyntaxNode { parse: tree.root(), index: tree.root_index() })
        .expect("missing root node")
}

syntax_node_simple!(Root);

impl Root {
    pub fn decls(self, tree: &Tree) -> impl DoubleEndedIterator<Item = Decl> + '_ {
        self.node().children(tree).filter_map(Decl::new)
    }
}

syntax_node_enum!(
    enum Decl {
        Alias(DeclAlias),
        Struct(DeclStruct),
        Const(DeclConst),
        ConstAssert(DeclConstAssert),
        Override(DeclOverride),
        Var(DeclVar),
        Fn(DeclFn),
    }
);

syntax_node_simple!(
    DeclAlias,
    struct DeclAliasData {
        attributes: AttributeList,
        alias_token: Token!(KeywordAlias),
        name: Token!(Identifier),
        equal_token: Token!(Equal),
        typ: TypeSpecifier,
        semi_token: Token!(SemiColon),
    }
);

syntax_node_simple!(
    DeclStruct,
    struct DeclStructData {
        attributes: AttributeList,
        struct_token: Token!(KeywordStruct),
        name: Token!(Identifier),
        fields: DeclStructFieldList,
    }
);

syntax_node_simple!(DeclStructFieldList);

impl DeclStructFieldList {
    pub fn fields(self, tree: &Tree) -> impl DoubleEndedIterator<Item = DeclStructField> + '_ {
        self.0.children(tree).filter_map(DeclStructField::new)
    }
}

syntax_node_simple!(
    DeclStructField,
    struct DeclStructFieldData {
        attributes: AttributeList,
        name: Token!(Identifier),
        colon_token: Token!(Colon),
        typ: TypeSpecifier,
    }
);

syntax_node_simple!(
    DeclConst,
    struct DeclConstData {
        const_token: Token!(KeywordConst),
        name: Token!(Identifier),
        colon_token: Token!(Colon),
        typ: TypeSpecifier,
        equal_token: Token!(Equal),
        value: Expression,
        semi_token: Token!(SemiColon),
    }
);

syntax_node_simple!(
    DeclConstAssert,
    struct DeclConstAssertData {
        assert_token: Token!(KeywordConstAssert),
        expr: Expression,
        semi_token: Token!(SemiColon),
    }
);

syntax_node_simple!(
    DeclOverride,
    struct DeclOverrideData {
        attributes: AttributeList,
        override_token: Token!(KeywordOverride),
        name: Token!(Identifier),
        colon_token: Token!(Colon),
        typ: TypeSpecifier,
        equal_token: Token!(Equal),
        value: Expression,
        semi_token: Token!(SemiColon),
    }
);

syntax_node_simple!(
    DeclVar,
    struct DeclVarData {
        attributes: AttributeList,
        var_token: Token!(KeywordVar),
        template: TemplateList,
        name: Token!(Identifier),
        colon_token: Token!(Colon),
        typ: TypeSpecifier,
        equal_token: Token!(Equal),
        value: Expression,
        semi_token: Token!(SemiColon),
    }
);

syntax_node_simple!(
    DeclFn,
    struct DeclFnData {
        attributes: AttributeList,
        fn_token: Token!(KeywordFn),
        name: Token!(Identifier),
        parameters: DeclFnParameterList,
        output: DeclFnOutput,
        body: StmtBlock,
    }
);
syntax_node_simple!(DeclFnParameterList);
syntax_node_simple!(
    DeclFnParameter,
    struct DeclFnParameterData {
        attributes: AttributeList,
        name: Token!(Identifier),
        colon_token: Token!(Colon),
        typ: TypeSpecifier,
    }
);
syntax_node_simple!(
    DeclFnOutput,
    struct DeclFnOutputData {
        arrow_token: Token!(ThinArrowRight),
        typ: TypeSpecifier,
    }
);

syntax_node_enum!(
    enum Statement {
        Assign(StmtAssign),
        Block(StmtBlock),
        Break(StmtBreak),
        Const(DeclConst),
        ConstAssert(DeclConstAssert),
        Continue(StmtContinue),
        Continuing(StmtContinuing),
        Decrement(StmtDecrement),
        Discard(StmtDiscard),
        Expr(StmtExpr),
        For(StmtFor),
        If(StmtIf),
        Increment(StmtIncrement),
        Let(StmtLet),
        Loop(StmtLoop),
        Return(StmtReturn),
        Switch(StmtSwitch),
        Var(DeclVar),
        While(StmtWhile),
    }
);

syntax_node_simple!(
    StmtAssign,
    struct StmtAssignData {
        lhs: Expression,
        op: AssignOp,
        rhs: Expression,
    }
);

syntax_node_enum!(
    enum AssignOp {
        Equal(Token!(Equal)),
        LessLessEqual(Token!(LessLessEqual)),
        GreaterGreaterEqual(Token!(GreaterGreaterEqual)),
        PlusEqual(Token!(PlusEqual)),
        MinusEqual(Token!(MinusEqual)),
        AsteriskEqual(Token!(AsteriskEqual)),
        SlashEqual(Token!(SlashEqual)),
        PercentEqual(Token!(PercentEqual)),
        ChevronEqual(Token!(ChevronEqual)),
        AmpersandEqual(Token!(AmpersandEqual)),
        BarEqual(Token!(BarEqual)),
    }
);

syntax_node_simple!(
    StmtBreak,
    struct StmtBreakData {
        break_token: Token!(KeywordBreak),
        if_token: Token!(KeywordIf),
        condition: Expression,
        semi_token: Token!(KeywordBreak),
    }
);
syntax_node_simple!(
    StmtContinue,
    struct StmtContinueData {
        continue_token: Token!(KeywordContinue),
        semi_token: Token!(KeywordBreak),
    }
);
syntax_node_simple!(
    StmtContinuing,
    struct StmtContinuingData {
        continuing_token: Token!(KeywordContinuing),
        stmt: Statement,
    }
);
syntax_node_simple!(
    StmtDecrement,
    struct StmtDecrementData {
        expr: Expression,
        decrement_token: Token!(MinusMinus),
    }
);
syntax_node_simple!(
    StmtDiscard,
    struct StmtDiscardData {
        continue_token: Token!(KeywordContinue),
        semi_token: Token!(KeywordBreak),
    }
);
syntax_node_simple!(
    StmtFor,
    struct StmtForData {
        for_token: Token!(KeywordFor),
        lparen_token: Token!(LParen),
        init: Statement,
        condition: Expression,
        condition_semi: Token!(SemiColon),
        post: Statement,
        rparen_token: Token!(RParen),
        body: Statement,
    }
);

syntax_node_simple!(StmtIf);

impl StmtIf {
    pub fn branches(self, tree: &Tree) -> impl DoubleEndedIterator<Item = StmtIfBranch> + '_ {
        self.0.children(tree).filter_map(StmtIfBranch::new)
    }
}

syntax_node_simple!(
    StmtIfBranch,
    struct StmtIfBranchData {
        else_token: Token!(KeywordElse),
        if_token: Token!(KeywordIf),
        condition: Expression,
        body: Statement,
    }
);

syntax_node_simple!(
    StmtIncrement,
    struct StmtIncrementData {
        expr: Expression,
        increment_token: Token!(PlusPlus),
    }
);

syntax_node_simple!(
    StmtLoop,
    struct StmtLoopData {
        loop_token: Token!(KeywordLoop),
        body: Statement,
    }
);

syntax_node_simple!(
    StmtReturn,
    struct StmtReturnData {
        return_token: Token!(KeywordReturn),
        expr: Expression,
        semi_token: Token!(SemiColon),
    }
);

syntax_node_simple!(
    StmtSwitch,
    struct StmtSwitchData {
        switch_token: Token!(KeywordSwitch),
        expr: Expression,
        lcurly_token: Token!(LCurly),
        branches: StmtSwitchBranchList,
        rcurly_token: Token!(RCurly),
    }
);

syntax_node_simple!(StmtSwitchBranchList);
impl StmtSwitchBranchList {
    pub fn cases(self, tree: &Tree) -> impl DoubleEndedIterator<Item = StmtSwitchBranch> + '_ {
        self.0.children(tree).filter_map(StmtSwitchBranch::new)
    }
}

syntax_node_simple!(
    StmtSwitchBranch,
    struct StmtSwitchBranchData {
        selector: StmtSwitchBranchSelector,
        colon_token: Token!(Colon),
        body: Statement,
    }
);

syntax_node_enum!(
    enum StmtSwitchBranchSelector {
        Default(Token!(KeywordDefault)),
        Case(StmtSwitchBranchCase),
    }
);

syntax_node_simple!(StmtSwitchBranchCase);
impl StmtSwitchBranchCase {
    pub fn patterns(
        self,
        tree: &Tree,
    ) -> impl DoubleEndedIterator<Item = StmtSwitchBranchCasePattern> + '_ {
        self.0.children(tree).filter_map(StmtSwitchBranchCasePattern::new)
    }
}

syntax_node_enum!(
    enum StmtSwitchBranchCasePattern {
        Default(Token!(KeywordDefault)),
        Expr(Expression),
    }
);

syntax_node_simple!(
    StmtWhile,
    struct StmtWhileData {
        while_token: Token!(KeywordWhile),
        expr: Expression,
        body: Statement,
    }
);

syntax_node_simple!(StmtBlock);
impl StmtBlock {
    pub fn statements(self, tree: &Tree) -> impl DoubleEndedIterator<Item = Statement> + '_ {
        self.0.children(tree).filter_map(Statement::new)
    }
}

syntax_node_simple!(
    StmtExpr,
    struct StmtExprData {
        expr: Expression,
        semi: Token!(SemiColon),
    }
);

syntax_node_simple!(
    StmtLet,
    struct StmtLetData {
        let_token: Token!(KeywordLet),
        name: Token!(Identifier),
        colon_token: Token!(Colon),
        typ: TypeSpecifier,
        equal_token: Token!(Equal),
        value: Expression,
        semi_token: Token!(SemiColon),
    }
);

syntax_node_enum!(
    enum Expression {
        Identifier(Token!(Identifier)),
        IdentifierWithTemplate(IdentifierWithTemplate),
        IntegerDecimal(Token!(IntegerDecimal)),
        IntegerHex(Token!(IntegerHex)),
        FloatDecimal(Token!(FloatDecimal)),
        FloatHex(Token!(FloatHex)),
        True(Token!(KeywordTrue)),
        False(Token!(KeywordFalse)),
        Call(ExprCall),
        Parens(ExprParens),
        Index(ExprIndex),
        Member(ExprMember),
        Prefix(ExprPrefix),
        Infix(ExprInfix),
    }
);

syntax_node_simple!(
    ExprInfix,
    struct ExprInfixData {
        lhs: Expression,
        op: InfixOp,
        rhs: Expression,
    }
);

syntax_node_enum!(
    enum InfixOp {
        Asterisk(Token!(Asterisk)),
        Slash(Token!(Slash)),
        Percent(Token!(Percent)),
        LessLess(Token!(LessLess)),
        GreaterGreater(Token!(GreaterGreater)),
        Chevron(Token!(Chevron)),
        Ampersand(Token!(Ampersand)),
        Bar(Token!(Bar)),
        Plus(Token!(Plus)),
        Minus(Token!(Minus)),
        Less(Token!(Less)),
        LessEqual(Token!(LessEqual)),
        Greater(Token!(Greater)),
        GreaterEqual(Token!(GreaterEqual)),
        EqualEqual(Token!(EqualEqual)),
        ExclamationEqual(Token!(ExclamationEqual)),
        AmpersandAmpersand(Token!(AmpersandAmpersand)),
        BarBar(Token!(BarBar)),
    }
);

syntax_node_simple!(ExprPrefix);

impl ExprPrefix {
    pub fn ops(self, tree: &Tree) -> impl DoubleEndedIterator<Item = UnaryOp> + '_ {
        self.0.children(tree).filter_map(UnaryOp::new)
    }

    pub fn expr(self, tree: &Tree) -> Option<Expression> {
        self.0.children(tree).next_back().and_then(Expression::new)
    }
}

syntax_node_enum!(
    enum UnaryOp {
        Minus(Token!(Minus)),
        Ampersand(Token!(Ampersand)),
        Asterisk(Token!(Asterisk)),
        Exclamation(Token!(Exclamation)),
        Tilde(Token!(Tilde)),
    }
);

syntax_node_simple!(
    ExprCall,
    struct ExprCallData {
        target: Expression,
        arguments: ArgumentList,
    }
);

syntax_node_simple!(ArgumentList);

impl ArgumentList {
    pub fn arguments(self, tree: &Tree) -> impl DoubleEndedIterator<Item = Argument> + '_ {
        self.0.children(tree).filter_map(Argument::new)
    }
}

syntax_node_simple!(
    Argument,
    struct ArgumentData {
        expr: Expression,
        comma: Token!(Comma),
    }
);

syntax_node_simple!(
    ExprParens,
    struct ExprParensData {
        lparen_token: Token!(LParen),
        value: Expression,
        rparen_token: Token!(RParen),
    }
);
syntax_node_simple!(
    ExprIndex,
    struct ExprIndexData {
        target: Expression,
        lbracket_token: Token!(LBracket),
        index: Expression,
        rbracket_token: Token!(RBracket),
    }
);
syntax_node_simple!(
    ExprMember,
    struct ExprMemberData {
        target: Expression,
        dot_token: Token!(Dot),
        member: Token!(Identifier),
    }
);

syntax_node_enum!(
    enum TypeSpecifier {
        Identifier(Token!(Identifier)),
        IdentifierWithTemplate(IdentifierWithTemplate),
    }
);

syntax_node_simple!(
    IdentifierWithTemplate,
    struct IdentifierWithTemplateData {
        name: Token!(Identifier),
        templates: TemplateList,
    }
);

syntax_node_simple!(TemplateList);

impl TemplateList {
    pub fn parameters(
        self,
        tree: &Tree,
    ) -> impl DoubleEndedIterator<Item = TemplateParameter> + '_ {
        self.0.children(tree).filter_map(TemplateParameter::new)
    }
}

syntax_node_simple!(
    TemplateParameter,
    struct TemplateParameterData {
        value: Expression,
        comma: Token!(Comma),
    }
);

syntax_node_simple!(AttributeList);

syntax_node_simple!(
    Attribute,
    struct AttributeData {
        at_token: Token!(AtSign),
        name: Token!(Identifier),
        arguments: ArgumentList,
    }
);
