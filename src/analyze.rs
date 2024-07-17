use std::{
    collections::{BTreeMap, HashMap},
    num::NonZeroU32,
};

use crate::{
    parse,
    syntax::{self, SyntaxNodeMatch},
};

static BUILTIN_FUNCTIONS: std::sync::OnceLock<wgsl_spec::FunctionInfo> = std::sync::OnceLock::new();

fn get_builtin_functions() -> &'static wgsl_spec::FunctionInfo {
    BUILTIN_FUNCTIONS.get_or_init(|| {
        wgsl_spec::include::functions().expect("could not load builtin function defintitions")
    })
}

pub type GlobalScope = HashMap<String, GlobalDeclaration>;

#[derive(Debug, Clone, Copy)]
pub struct GlobalDeclaration {
    pub node: parse::NodeIndex,
}

impl GlobalDeclaration {
    pub fn name(&self, tree: &parse::Tree) -> Option<syntax::Token!(Identifier)> {
        syntax::Decl::from_tree(tree, self.node).unwrap().name(tree)
    }
}

/// Found a declaration that conflicts with the name of another declaration.
#[derive(Debug)]
pub struct ErrorDuplicate {
    pub conflicts: Vec<GlobalDeclaration>,
}

pub fn collect_global_scope(
    tree: &parse::Tree,
    source: &str,
) -> (GlobalScope, BTreeMap<String, ErrorDuplicate>) {
    let root = syntax::root(tree);

    let mut scope = GlobalScope::with_capacity(root.parse_node().children().len());
    let mut errors = BTreeMap::new();

    for decl in root.decls(tree) {
        let (name, index) = match decl {
            syntax::Decl::Alias(alias) => (alias.extract(tree).name, alias.index()),
            syntax::Decl::Struct(strukt) => (strukt.extract(tree).name, strukt.index()),
            syntax::Decl::Const(konst) => (konst.extract(tree).name, konst.index()),
            syntax::Decl::Override(overide) => (overide.extract(tree).name, overide.index()),
            syntax::Decl::Var(war) => (war.extract(tree).name, war.index()),
            syntax::Decl::Fn(func) => (func.extract(tree).name, func.index()),
        };

        if let Some(name) = name {
            let text = &source[name.byte_range()];
            let decl = GlobalDeclaration { node: index };
            if let Some(previous) = scope.insert(text.into(), decl) {
                errors
                    .entry(text.into())
                    .or_insert_with(|| ErrorDuplicate { conflicts: vec![previous] })
                    .conflicts
                    .push(decl);
            }
        }
    }

    (scope, errors)
}

pub struct Context<'a> {
    builtin_functions: &'static wgsl_spec::FunctionInfo,
    global_scope: &'a GlobalScope,
    tree: &'a parse::Tree,
    source: &'a str,

    scope: Scope<'a>,

    errors: Vec<Error>,

    /// identifiers which have had their reference resolved.
    references: HashMap<parse::NodeIndex, Reference>,

    /// If set, we should record a capture of the visible symbols during name resolution.
    capture: Option<CaptureSymbols<'a>>,
}

struct CaptureSymbols<'a> {
    /// At which node we should perform the capture.
    node: parse::NodeIndex,
    /// The references we captured during name resolution.
    references: Vec<(&'a str, Reference)>,
}

#[derive(Debug)]
enum Reference {
    Node(parse::NodeIndex),
    BuiltinFunction(&'static wgsl_spec::Function),
}

#[derive(Debug)]
pub enum Error {
    UnresolvedReference(ErrorUnresolvedReference),
}

#[derive(Debug)]
pub struct ErrorUnresolvedReference {
    pub node: parse::NodeIndex,
}

impl<'a> Context<'a> {
    pub fn new(global_scope: &'a GlobalScope, tree: &'a parse::Tree, source: &'a str) -> Self {
        Self {
            builtin_functions: get_builtin_functions(),
            global_scope,
            tree,
            source,
            scope: Scope::new(),
            errors: Vec::new(),
            references: HashMap::new(),

            capture: None,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.scope.reset();
        self.references.clear();
        self.capture.take();
        self.errors.clear();
    }

    pub fn analyze_decl(&mut self, decl: syntax::Decl) -> &mut Vec<Error> {
        // name resolution:
        {
            let scope_root = self.scope.begin();
            self.resolve_references(decl.index());
            self.scope.end(scope_root);
        }

        &mut self.errors
    }

    pub fn get_node_symbol(&self, node: parse::NodeIndex) -> Option<ResolvedSymbol> {
        let reference = self.references.get(&node)?;
        let kind = match reference {
            Reference::Node(node) => ResolvedSymbolKind::User(*node),
            Reference::BuiltinFunction(function) => ResolvedSymbolKind::BuiltinFunction(function),
        };
        Some(ResolvedSymbol { name: self.source[self.tree.node(node).byte_range()].into(), kind })
    }

    pub fn capture_symbols_at(&mut self, node: parse::NodeIndex) {
        self.capture = Some(CaptureSymbols { node, references: Vec::new() });
    }

    pub fn get_captured_symbols(&self) -> Vec<ResolvedSymbol> {
        let Some(capture) = &self.capture else { return Vec::new() };
        let mut symbols = Vec::with_capacity(capture.references.len());

        for (name, reference) in capture.references.iter() {
            symbols.push(ResolvedSymbol {
                name: (*name).into(),
                kind: match reference {
                    Reference::Node(node) => ResolvedSymbolKind::User(*node),
                    Reference::BuiltinFunction(function) => {
                        ResolvedSymbolKind::BuiltinFunction(function)
                    },
                },
            });
        }

        symbols
    }

    #[cold]
    fn capture_symbols(&mut self) {
        let Some(capture) = &mut self.capture else { return };
        for (name, node) in self.scope.iter_symbols() {
            capture.references.push((name, Reference::Node(node)));
        }

        for (name, decl) in self.global_scope.iter() {
            capture.references.push((name, Reference::Node(decl.node)))
        }

        for (name, function) in self.builtin_functions.functions.iter() {
            capture.references.push((name, Reference::BuiltinFunction(function)))
        }
    }

    fn resolve_references(&mut self, index: parse::NodeIndex) {
        if let Some(capture) = &mut self.capture {
            if index == capture.node {
                self.capture_symbols();
            }
        }

        let node = self.tree.node(index);
        match node.tag() {
            parse::Tag::Identifier => {
                let name = &self.source[node.byte_range()];

                if let Some(local) = self.scope.get(name) {
                    self.references.insert(index, Reference::Node(local));
                } else if let Some(global) = self.global_scope.get(name) {
                    self.references.insert(index, Reference::Node(global.node));
                } else if let Some(builtin) = self.builtin_functions.functions.get(name) {
                    self.references.insert(index, Reference::BuiltinFunction(builtin));
                } else {
                    self.errors
                        .push(Error::UnresolvedReference(ErrorUnresolvedReference { node: index }));
                }
            },

            parse::Tag::Attribute => {
                // don't attempt to resolve the attribute name
                let syntax_node = syntax::SyntaxNode::new(node, index);
                let attribute = syntax::Attribute::new(syntax_node).unwrap();

                // TODO: introduce context-dependent names
                if let Some(arguments) = attribute.extract(self.tree).arguments {
                    self.resolve_references(arguments.index());
                }
            },

            parse::Tag::ExprMember => {
                // members cannot be resolved until types are known, so ignore this for now
            },

            parse::Tag::StmtBlock => {
                let block_scope = self.scope.begin();
                for child in node.children() {
                    self.resolve_references(child);
                }
                self.scope.end(block_scope);
            },

            parse::Tag::DeclConst => {
                let konst = syntax::DeclConst::new(syntax::SyntaxNode::new(node, index)).unwrap();
                let fields = konst.extract(self.tree);
                self.resolve_variable_declaration(index, fields.name, fields.typ, fields.value);
            },
            parse::Tag::DeclVar => {
                let war = syntax::DeclVar::new(syntax::SyntaxNode::new(node, index)).unwrap();
                let fields = war.extract(self.tree);
                self.resolve_variable_declaration(index, fields.name, fields.typ, fields.value);
            },
            parse::Tag::StmtLet => {
                let lett = syntax::StmtLet::new(syntax::SyntaxNode::new(node, index)).unwrap();
                let fields = lett.extract(self.tree);
                self.resolve_variable_declaration(index, fields.name, fields.typ, fields.value);
            },

            parse::Tag::DeclFnParameter => {
                let param =
                    syntax::DeclFnParameter::new(syntax::SyntaxNode::new(node, index)).unwrap();
                let fields = param.extract(self.tree);
                self.resolve_variable_declaration(index, fields.name, fields.typ, None);
            },

            tag => {
                if tag.is_syntax() {
                    for child in node.children() {
                        self.resolve_references(child);
                    }
                }
            },
        }
    }

    fn resolve_variable_declaration(
        &mut self,
        index: parse::NodeIndex,
        name: Option<syntax::Token!(Identifier)>,
        typ: Option<syntax::TypeSpecifier>,
        value: Option<syntax::Expression>,
    ) {
        if let Some(name) = name {
            let text = &self.source[name.byte_range()];
            self.scope.insert(text, index);
            self.references.insert(name.index(), Reference::Node(index));
        }
        self.resolve_references_maybe(typ);
        self.resolve_references_maybe(value);
    }

    fn resolve_references_maybe(&mut self, index: Option<impl syntax::SyntaxNodeMatch>) {
        if let Some(index) = index {
            self.resolve_references(index.index());
        }
    }
}

struct Scope<'a> {
    symbols: HashMap<&'a str, ScopeSymbol>,
    next_scope: ScopeId,
    active_scopes: Vec<ScopeId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ScopeId(u32);

struct ScopeSymbol {
    /// Which node this symbol refers to.
    node: parse::NodeIndex,
    /// In which scope this symbol is valid.
    scope: ScopeId,
    /// Points to a shadowed symbol of the same name.
    shadowed: Option<Box<ScopeSymbol>>,
}

impl<'a> Scope<'a> {
    pub fn new() -> Scope<'static> {
        Scope { symbols: HashMap::new(), next_scope: ScopeId(0), active_scopes: Vec::new() }
    }

    fn reset(&mut self) {
        self.symbols.clear();
        self.active_scopes.clear();
        self.next_scope = ScopeId(0);
    }

    pub fn begin(&mut self) -> ScopeId {
        let scope = self.next_scope;
        self.active_scopes.push(scope);
        self.next_scope.0 += 1;
        scope
    }

    pub fn end(&mut self, scope: ScopeId) {
        assert_eq!(Some(scope), self.active_scopes.last().copied());
        self.active_scopes.pop();
    }

    pub fn insert(&mut self, name: &'a str, node: parse::NodeIndex) {
        let new = ScopeSymbol { node, scope: *self.active_scopes.last().unwrap(), shadowed: None };
        match self.symbols.entry(name) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let shadowed = std::mem::replace(entry.get_mut(), new);
                entry.get_mut().shadowed = Some(Box::new(shadowed));
            },
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(new);
            },
        }
    }

    pub fn get(&mut self, name: &'a str) -> Option<parse::NodeIndex> {
        use std::collections::hash_map::Entry;
        let Entry::Occupied(mut entry) = self.symbols.entry(name) else { return None };

        match Self::get_slot(entry.get_mut(), &self.active_scopes) {
            Some(node) => Some(node),
            None => {
                entry.remove();
                None
            },
        }
    }

    fn get_slot(slot: &mut ScopeSymbol, active_scopes: &[ScopeId]) -> Option<parse::NodeIndex> {
        loop {
            if Self::is_active(slot.scope, active_scopes) {
                return Some(slot.node);
            }
            *slot = *slot.shadowed.take()?;
        }
    }

    fn is_active(target: ScopeId, scopes: &[ScopeId]) -> bool {
        if scopes.len() < 16 {
            scopes.iter().rev().any(|x| *x == target)
        } else {
            scopes.binary_search(&target).is_ok()
        }
    }

    fn iter_symbols(&mut self) -> impl Iterator<Item = (&'a str, parse::NodeIndex)> + '_ {
        self.symbols.iter_mut().filter_map(|(name, slot)| {
            let node = Self::get_slot(slot, &self.active_scopes)?;
            Some((*name, node))
        })
    }
}

#[derive(Debug)]
pub struct ResolvedSymbol {
    pub name: String,
    pub kind: ResolvedSymbolKind,
}

#[derive(Debug)]
pub enum ResolvedSymbolKind {
    User(parse::NodeIndex),
    BuiltinFunction(&'static wgsl_spec::Function),
}

#[derive(Debug)]
pub enum Type {
    Unknown,
    Scalar(TypeScalar),
    Vec(u8, TypeScalar),
    Mat(u8, u8, TypeScalar),
    Atomic(TypeScalar),
    Array(Box<Type>, Option<NonZeroU32>),
    Struct(Box<TypeStruct>),
}

const _: () = assert!(std::mem::size_of::<Type>() <= 16);

#[derive(Debug)]
pub struct TypeStruct {
    pub name: String,
    pub fields: Vec<TypeStructField>,
}

#[derive(Debug)]
pub struct TypeStructField {
    pub name: String,
    pub typ: Type,
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Unknown => write!(f, "?"),

            Type::Scalar(scalar) => scalar.fmt(f),

            Type::Vec(dims, scalar) => write!(f, "vec{dims}<{scalar}>"),

            Type::Mat(cols, rows, scalar) => write!(f, "mat{cols}x{rows}<{scalar}>"),

            Type::Atomic(scalar) => write!(f, "atomic<{scalar}>"),

            Type::Array(inner, None) => write!(f, "array<{inner}>"),
            Type::Array(inner, Some(count)) => write!(f, "array<{inner}, {count}>"),

            Type::Struct(strukt) => write!(f, "{}", strukt.name),
        }
    }
}

#[derive(Debug)]
pub enum TypeScalar {
    I32,
    U32,
    F32,
    F16,
    Bool,
    AbstractInt,
    AbstractFloat,
}

impl std::fmt::Display for TypeScalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeScalar::I32 => write!(f, "i32"),
            TypeScalar::U32 => write!(f, "u32"),
            TypeScalar::F32 => write!(f, "f32"),
            TypeScalar::F16 => write!(f, "f16"),
            TypeScalar::Bool => write!(f, "bool"),
            TypeScalar::AbstractInt => write!(f, "AbstractInt"),
            TypeScalar::AbstractFloat => write!(f, "AbstractFloat"),
        }
    }
}
