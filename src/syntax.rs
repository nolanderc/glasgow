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

    pub fn children(self, tree: &Tree) -> impl Iterator<Item = SyntaxNode> + '_ {
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
        self.node().parse
    }

    fn index(self) -> parse::NodeIndex {
        self.node().index
    }

    fn from_tree(tree: &Tree, index: parse::NodeIndex) -> Option<Self> {
        Self::new(SyntaxNode { parse: tree.node(index), index })
    }
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
        pub struct $extracted {
            $( pub $field: Option<$fieldty> ),*
        }

        #[allow(dead_code)]
        impl $name {
            pub fn extract(self, tree: &Tree) -> $extracted {
                let mut children = self.0.children(tree).peekable();
                $( let $field = try_extract::<$fieldty>(&mut children); )*
                $extracted { $( $field ),* }
            }
        }
    };
}

#[inline(always)]
fn try_extract<T: SyntaxNodeMatch>(
    nodes: &mut std::iter::Peekable<impl Iterator<Item = SyntaxNode>>,
) -> Option<T> {
    loop {
        let node = *nodes.peek()?;

        if let Some(x) = T::new(node) {
            nodes.next();
            return Some(x);
        }

        if matches!(node.parse.tag(), parse::Tag::InvalidToken | parse::Tag::InvalidSyntax) {
            nodes.next();
        } else {
            return None;
        }
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
pub struct Token<const TAG: u8>(SyntaxNode);

impl<const TAG: u8> SyntaxNodeMatch for Token<TAG> {
    fn new(node: SyntaxNode) -> Option<Self> {
        if node.parse.tag() as u8 == TAG {
            Some(Token(node))
        } else {
            None
        }
    }

    fn node(self) -> SyntaxNode {
        self.0
    }
}

impl<const TAG: u8> Token<TAG> {
    pub fn byte_range(self) -> std::ops::Range<usize> {
        self.0.parse.byte_range()
    }
}

macro_rules! Token {
    ($tag:ident) => {
        Token<{ parse::Tag::$tag as u8 }>
    };
}

pub fn root(tree: &Tree) -> Root {
    Root::new(SyntaxNode { parse: tree.root(), index: tree.root_index() })
        .expect("missing root node")
}

syntax_node_simple!(Root);

impl Root {
    pub fn decls(self, tree: &Tree) -> impl Iterator<Item = Decl> + '_ {
        self.node().children(tree).filter_map(Decl::new)
    }
}

syntax_node_enum!(
    enum Decl {
        Alias(DeclAlias),
        Struct(DeclStruct),
        Const(DeclConst),
        Override(DeclOverride),
        Var(DeclVar),
        Fn(DeclFn),
    }
);

syntax_node_simple!(
    DeclAlias,
    struct DeclAliasFields {
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
    struct DeclStructFields {
        attributes: AttributeList,
        struct_token: Token!(KeywordAlias),
        name: Token!(Identifier),
        fields: DeclStructFieldList,
    }
);
syntax_node_simple!(DeclStructFieldList);

syntax_node_simple!(
    DeclConst,
    struct DeclConstFields {
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
    DeclOverride,
    struct DeclOverrideFields {
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
    struct DeclVarFields {
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
    struct DeclFnFields {
        attributes: AttributeList,
        fn_token: Token!(KeywordFn),
        name: Token!(Identifier),
        parameters: DeclFnParameterList,
        output: DeclFnOutput,
        body: StmtBlock,
    }
);
syntax_node_simple!(DeclFnParameterList);
syntax_node_simple!(DeclFnOutput);

syntax_node_simple!(StmtBlock);

syntax_node_enum!(
    enum Expression {
        Identifier(Token!(Identifier)),
        IdentifierWithTemplate(IdentifierWithTemplate),
    }
);

syntax_node_simple!(ArgumentList);

syntax_node_enum!(
    enum TypeSpecifier {
        Identifier(Token!(Identifier)),
        IdentifierWithTemplate(IdentifierWithTemplate),
    }
);

syntax_node_simple!(IdentifierWithTemplate);

syntax_node_simple!(TemplateList);
syntax_node_simple!(AttributeList);

syntax_node_simple!(
    Attribute,
    struct AttributeFields {
        at_token: Token!(AtSign),
        name: Token!(Identifier),
        arguments: ArgumentList,
    }
);
