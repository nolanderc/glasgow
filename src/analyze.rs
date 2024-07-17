use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, HashMap},
    num::NonZeroU32,
    rc::Rc,
};

use crate::{
    parse,
    syntax::{self, SyntaxNodeMatch},
};

static BUILTIN_FUNCTIONS: std::sync::OnceLock<wgsl_spec::FunctionInfo> = std::sync::OnceLock::new();
static BUILTIN_TOKENS: std::sync::OnceLock<wgsl_spec::TokenInfo> = std::sync::OnceLock::new();

fn get_builtin_functions() -> &'static wgsl_spec::FunctionInfo {
    BUILTIN_FUNCTIONS.get_or_init(|| {
        wgsl_spec::include::functions().expect("could not load builtin function defintitions")
    })
}

fn get_builtin_tokens() -> &'static wgsl_spec::TokenInfo {
    BUILTIN_TOKENS.get_or_init(|| {
        wgsl_spec::include::tokens().expect("could not load builtin token defintitions")
    })
}

pub type GlobalScope = HashMap<String, GlobalDeclaration>;

#[derive(Debug, Clone, Copy)]
pub struct GlobalDeclaration {
    pub node: ReferenceNode,
}

impl GlobalDeclaration {
    pub fn name(&self, tree: &parse::Tree) -> Option<syntax::Token!(Identifier)> {
        self.node.name(tree)
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
        let node = ReferenceNode::from(decl);
        if let Some(name) = node.name(tree) {
            let text = &source[name.byte_range()];
            let global = GlobalDeclaration { node };
            if let Some(previous) = scope.insert(text.into(), global) {
                errors
                    .entry(text.into())
                    .or_insert_with(|| ErrorDuplicate { conflicts: vec![previous] })
                    .conflicts
                    .push(global);
            }
        }
    }

    (scope, errors)
}

pub struct Context<'a> {
    builtin_functions: &'static wgsl_spec::FunctionInfo,
    builtin_tokens: &'static wgsl_spec::TokenInfo,

    global_scope: &'a GlobalScope,
    tree: &'a parse::Tree,
    source: &'a str,

    scope: Scope<'a>,

    errors: RefCell<Vec<Error>>,

    /// identifiers which have had their reference resolved.
    references: RefCell<HashMap<parse::NodeIndex, Reference>>,

    /// The inferred type for each syntax node.
    types: Cell<HashMap<parse::NodeIndex, Option<Type>>>,

    /// If set, we should record a capture of the visible symbols during name resolution.
    /// At which node we should perform the capture.
    capture_node: Option<parse::NodeIndex>,
    /// The references we captured during name resolution.
    capture: CaptureSymbols<'a>,
}

#[derive(Default)]
struct CaptureSymbols<'a> {
    references: Vec<(&'a str, Reference)>,
}

#[derive(Debug, Clone)]
pub enum Reference {
    User(ReferenceNode),
    BuiltinFunction(&'static wgsl_spec::Function),
    BuiltinTypeAlias(&'static String, &'static String),
    Type(Type),
}

impl Reference {
    pub fn name(&self, tree: &parse::Tree) -> Option<syntax::Token!(Identifier)> {
        match self {
            &Reference::User(node) => node.name(tree),
            Reference::Type(typ) => match typ {
                Type::Struct(strukt) => strukt.extract(tree).name,
                Type::Fn(func) => func.extract(tree).name,
                _ => None,
            },
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ReferenceNode {
    Alias(syntax::DeclAlias),
    Struct(syntax::DeclStruct),
    StructField(syntax::DeclStructField),
    Const(syntax::DeclConst),
    Override(syntax::DeclOverride),
    Var(syntax::DeclVar),
    Fn(syntax::DeclFn),
    FnParameter(syntax::DeclFnParameter),
    Let(syntax::StmtLet),
}

impl From<syntax::Decl> for ReferenceNode {
    fn from(decl: syntax::Decl) -> ReferenceNode {
        match decl {
            syntax::Decl::Alias(x) => ReferenceNode::Alias(x),
            syntax::Decl::Struct(x) => ReferenceNode::Struct(x),
            syntax::Decl::Const(x) => ReferenceNode::Const(x),
            syntax::Decl::Override(x) => ReferenceNode::Override(x),
            syntax::Decl::Var(x) => ReferenceNode::Var(x),
            syntax::Decl::Fn(x) => ReferenceNode::Fn(x),
        }
    }
}

impl ReferenceNode {
    pub fn raw(self) -> syntax::SyntaxNode {
        match self {
            ReferenceNode::Fn(x) => x.node(),
            ReferenceNode::Alias(x) => x.node(),
            ReferenceNode::Struct(x) => x.node(),
            ReferenceNode::StructField(x) => x.node(),
            ReferenceNode::Const(x) => x.node(),
            ReferenceNode::Override(x) => x.node(),
            ReferenceNode::Var(x) => x.node(),
            ReferenceNode::FnParameter(x) => x.node(),
            ReferenceNode::Let(x) => x.node(),
        }
    }

    pub fn name(self, tree: &parse::Tree) -> Option<syntax::Token!(Identifier)> {
        match self {
            ReferenceNode::Alias(x) => x.extract(tree).name,
            ReferenceNode::Struct(x) => x.extract(tree).name,
            ReferenceNode::StructField(x) => x.extract(tree).name,
            ReferenceNode::Const(x) => x.extract(tree).name,
            ReferenceNode::Override(x) => x.extract(tree).name,
            ReferenceNode::Var(x) => x.extract(tree).name,
            ReferenceNode::Fn(x) => x.extract(tree).name,
            ReferenceNode::FnParameter(x) => x.extract(tree).name,
            ReferenceNode::Let(x) => x.extract(tree).name,
        }
    }
}

#[derive(Debug)]
pub enum Error {
    UnresolvedReference(ErrorUnresolvedReference),
    InvalidCallTarget(syntax::ExprCall, Type),
    InvalidIndexTarget(syntax::ExprIndex, Type),
    InvalidIndexIndex(syntax::ExprIndex, Type),
    InvalidMember(syntax::ExprMember, Type),
}

#[derive(Debug)]
pub struct ErrorUnresolvedReference {
    pub node: syntax::Token!(Identifier),
}

impl<'a> Context<'a> {
    pub fn new(global_scope: &'a GlobalScope, tree: &'a parse::Tree, source: &'a str) -> Self {
        Self {
            builtin_functions: get_builtin_functions(),
            builtin_tokens: get_builtin_tokens(),

            global_scope,
            tree,
            source,
            scope: Scope::new(),
            errors: Vec::new().into(),
            references: Default::default(),
            types: Default::default(),

            capture_node: None,
            capture: Default::default(),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.scope.reset();
        self.references = Default::default();
        self.capture = Default::default();
        self.errors.get_mut().clear();
    }

    pub fn analyze_decl(&mut self, decl: syntax::Decl) -> &mut Vec<Error> {
        // name resolution:
        let scope_root = self.scope.begin();
        self.resolve_references(decl.index());
        self.scope.end(scope_root);

        match decl {
            syntax::Decl::Alias(_) => {},
            syntax::Decl::Struct(_) => {},
            syntax::Decl::Const(konst) => {
                let data = konst.extract(self.tree);
                self.check_variable_decl_types(data.typ, data.value);
            },
            syntax::Decl::Override(overide) => {
                let data = overide.extract(self.tree);
                self.check_variable_decl_types(data.typ, data.value);
            },
            syntax::Decl::Var(war) => {
                let data = war.extract(self.tree);
                self.check_variable_decl_types(data.typ, data.value);
            },
            syntax::Decl::Fn(func) => {
                let data = func.extract(self.tree);
                if let Some(body) = data.body {
                    self.check_stmt_types(syntax::Statement::Block(body))
                }
            },
        }

        self.errors.get_mut()
    }

    pub fn get_node_symbol(&self, node: parse::NodeIndex) -> Option<ResolvedSymbol> {
        let reference = self.references.borrow().get(&node)?.clone();
        let typ = self.type_of_reference(&reference).or_else(|| self.get_inferred_type(node));
        Some(ResolvedSymbol {
            name: self.source[self.tree.node(node).byte_range()].into(),
            reference,
            typ,
        })
    }

    pub fn capture_symbols_at(&mut self, node: parse::NodeIndex) {
        self.capture_node = Some(node);
    }

    pub fn get_captured_symbols(&self) -> Vec<ResolvedSymbol> {
        let mut symbols = Vec::with_capacity(self.capture.references.len());

        for (name, reference) in self.capture.references.iter() {
            symbols.push(ResolvedSymbol {
                name: (*name).into(),
                reference: reference.clone(),
                typ: self.type_of_reference(reference),
            });
        }

        symbols
    }

    #[cold]
    fn capture_symbols(&mut self) -> CaptureSymbols<'a> {
        let mut references = Vec::new();

        for (name, node) in self.scope.iter_symbols() {
            references.push((name, Reference::User(node)));
        }

        for (name, decl) in self.global_scope.iter() {
            references.push((name, Reference::User(decl.node)))
        }

        for (name, function) in self.builtin_functions.functions.iter() {
            references.push((name, Reference::BuiltinFunction(function)))
        }

        for (name, original) in self.builtin_tokens.type_aliases.iter() {
            references.push((name, Reference::BuiltinTypeAlias(name, original)))
        }

        for name in self.builtin_tokens.primitive_types.iter() {
            let Some(typ) = self.parse_type_specifier(name) else {
                unreachable!("could parse primitive type: {name}")
            };
            references.push((name, Reference::Type(typ)));
        }

        for name in self.builtin_tokens.type_generators.iter() {
            let Some(typ) = self.parse_type_specifier(name) else {
                unreachable!("could parse type generator: {name}")
            };
            references.push((name, Reference::Type(typ)));
        }

        CaptureSymbols { references }
    }

    fn resolve_references(&mut self, index: parse::NodeIndex) {
        if Some(index) == self.capture_node {
            self.capture = self.capture_symbols();
        }

        let node = self.tree.node(index);
        match node.tag() {
            parse::Tag::Identifier => {
                let name = &self.source[node.byte_range()];

                let reference = if let Some(local) = self.scope.get(name) {
                    Reference::User(local)
                } else if let Some(global) = self.global_scope.get(name) {
                    Reference::User(global.node)
                } else if let Some(builtin) = self.builtin_functions.functions.get(name) {
                    Reference::BuiltinFunction(builtin)
                } else if let Some((name, original)) =
                    self.builtin_tokens.type_aliases.get_key_value(name)
                {
                    Reference::BuiltinTypeAlias(name, original)
                } else {
                    self.errors.get_mut().push(Error::UnresolvedReference(
                        ErrorUnresolvedReference {
                            node: <syntax::Token!(Identifier)>::new(syntax::SyntaxNode::new(
                                node, index,
                            ))
                            .unwrap(),
                        },
                    ));
                    return;
                };

                self.references.get_mut().insert(index, reference);
            },

            parse::Tag::Attribute => {
                // don't attempt to resolve the attribute name
                let syntax_node = syntax::SyntaxNode::new(node, index);
                let attribute = syntax::Attribute::new(syntax_node).unwrap();

                // TODO: introduce context-dependent names
                if let Some(_arguments) = attribute.extract(self.tree).arguments {
                    // self.resolve_references(arguments.index());
                }
            },

            parse::Tag::ExprMember => {
                let konst = syntax::ExprMember::new(syntax::SyntaxNode::new(node, index)).unwrap();
                let data = konst.extract(self.tree);
                self.resolve_references_maybe(data.target);
                // the members themselves cannot be resolved until types are known
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
                let data = konst.extract(self.tree);
                self.resolve_variable_declaration(
                    ReferenceNode::Const(konst),
                    data.name,
                    data.typ,
                    data.value,
                );
            },
            parse::Tag::DeclVar => {
                let war = syntax::DeclVar::new(syntax::SyntaxNode::new(node, index)).unwrap();
                let data = war.extract(self.tree);
                self.resolve_variable_declaration(
                    ReferenceNode::Var(war),
                    data.name,
                    data.typ,
                    data.value,
                );
            },
            parse::Tag::StmtLet => {
                let lett = syntax::StmtLet::new(syntax::SyntaxNode::new(node, index)).unwrap();
                let data = lett.extract(self.tree);
                self.resolve_variable_declaration(
                    ReferenceNode::Let(lett),
                    data.name,
                    data.typ,
                    data.value,
                );
            },

            parse::Tag::DeclStructField => {
                let field =
                    syntax::DeclStructField::new(syntax::SyntaxNode::new(node, index)).unwrap();
                let data = field.extract(self.tree);
                self.resolve_references_maybe(data.attributes);
                if let Some(name) = data.name {
                    self.references
                        .get_mut()
                        .insert(name.index(), Reference::User(ReferenceNode::StructField(field)));
                }
                self.resolve_references_maybe(data.typ);
            },

            parse::Tag::DeclFnParameter => {
                let param =
                    syntax::DeclFnParameter::new(syntax::SyntaxNode::new(node, index)).unwrap();
                let data = param.extract(self.tree);
                self.resolve_variable_declaration(
                    ReferenceNode::FnParameter(param),
                    data.name,
                    data.typ,
                    None,
                );
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

    fn resolve_references_maybe(&mut self, index: Option<impl syntax::SyntaxNodeMatch>) {
        if let Some(index) = index {
            self.resolve_references(index.index());
        }
    }

    fn resolve_variable_declaration(
        &mut self,
        node: ReferenceNode,
        name: Option<syntax::Token!(Identifier)>,
        typ: Option<syntax::TypeSpecifier>,
        value: Option<syntax::Expression>,
    ) {
        if let Some(name) = name {
            let text = &self.source[name.byte_range()];
            self.scope.insert(text, node);
            self.references.get_mut().insert(name.index(), Reference::User(node));
        }
        self.resolve_references_maybe(typ);
        self.resolve_references_maybe(value);
    }

    fn check_variable_decl_types(
        &self,
        _typ: Option<syntax::TypeSpecifier>,
        value: Option<syntax::Expression>,
    ) {
        // TODO: ensure the type of the expression is valid
        self.infer_type_expr_maybe(value);
    }

    fn check_stmt_types(&self, stmt: syntax::Statement) {
        match stmt {
            syntax::Statement::Block(block) => {
                for stmt in block.statements(self.tree) {
                    self.check_stmt_types(stmt);
                }
            },
            syntax::Statement::Expr(expr) => {
                let data = expr.extract(self.tree);
                self.infer_type_expr_maybe(data.expr);
            },
            syntax::Statement::Const(konst) => {
                let data = konst.extract(self.tree);
                self.check_variable_decl_types(data.typ, data.value);
            },
            syntax::Statement::Var(war) => {
                let data = war.extract(self.tree);
                self.check_variable_decl_types(data.typ, data.value);
            },
            syntax::Statement::Let(lett) => {
                let data = lett.extract(self.tree);
                self.check_variable_decl_types(data.typ, data.value);
            },
        }
    }

    fn infer_type_expr(&self, expr: syntax::Expression) -> Option<Type> {
        let mut cache = self.types.take();

        let typ = match cache.entry(expr.index()) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.get().as_ref().cloned(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(self.infer_type_expr_uncached(expr)).as_ref().cloned()
            },
        };

        self.types.set(cache);

        typ
    }

    fn get_inferred_type(&self, node: parse::NodeIndex) -> Option<Type> {
        let cache = self.types.take();
        let typ = cache.get(&node).cloned().flatten();
        self.types.set(cache);
        typ
    }

    fn infer_type_expr_maybe(&self, expr: Option<syntax::Expression>) -> Option<Type> {
        self.infer_type_expr(expr?)
    }

    fn infer_type_expr_uncached(&self, expr: syntax::Expression) -> Option<Type> {
        match expr {
            syntax::Expression::Identifier(name) => {
                self.type_of_reference(self.references.borrow().get(&name.index())?)
            },
            syntax::Expression::IdentifierWithTemplate(identifier) => self
                .type_from_specifier(
                    syntax::TypeSpecifier::IdentifierWithTemplate(identifier),
                    self.tree,
                    self.source,
                )
                .map(|x| Type::Type(Rc::new(x))),
            syntax::Expression::IntegerDecimal(number) => {
                match self.source[number.byte_range()].chars().next_back() {
                    Some('u') => Some(Type::Scalar(TypeScalar::U32)),
                    Some('i') => Some(Type::Scalar(TypeScalar::I32)),
                    _ => Some(Type::Scalar(TypeScalar::AbstractInt)),
                }
            },
            syntax::Expression::IntegerHex(number) => {
                match self.source[number.byte_range()].chars().next_back() {
                    Some('u') => Some(Type::Scalar(TypeScalar::U32)),
                    Some('i') => Some(Type::Scalar(TypeScalar::I32)),
                    _ => Some(Type::Scalar(TypeScalar::AbstractInt)),
                }
            },
            syntax::Expression::FloatDecimal(number) => {
                match self.source[number.byte_range()].chars().next_back() {
                    Some('f') => Some(Type::Scalar(TypeScalar::F32)),
                    Some('h') => Some(Type::Scalar(TypeScalar::F16)),
                    _ => Some(Type::Scalar(TypeScalar::AbstractInt)),
                }
            },
            syntax::Expression::FloatHex(number) => {
                match self.source[number.byte_range()].chars().next_back() {
                    Some('f') => Some(Type::Scalar(TypeScalar::F32)),
                    Some('h') => Some(Type::Scalar(TypeScalar::F16)),
                    _ => Some(Type::Scalar(TypeScalar::AbstractInt)),
                }
            },

            syntax::Expression::True(_) | syntax::Expression::False(_) => {
                Some(Type::Scalar(TypeScalar::Bool))
            },
            syntax::Expression::Call(call) => {
                let data = call.extract(self.tree);
                let target_type = self.infer_type_expr_maybe(data.target);

                if let Some(arguments) = data.arguments {
                    for argument in arguments.arguments(self.tree) {
                        // TODO: check that the type is valid for the given argument
                        self.infer_type_expr(argument);
                    }
                }

                match target_type? {
                    Type::Type(typ) => Some(Type::clone(&typ)),
                    Type::Fn(func) => {
                        let data = func.extract(self.tree);
                        let spec = data.output?.extract(self.tree).typ?;
                        self.type_from_specifier(spec, self.tree, self.source)
                    },
                    typ => {
                        self.errors.borrow_mut().push(Error::InvalidCallTarget(call, typ));
                        None
                    },
                }
            },

            syntax::Expression::Parens(parens) => {
                let data = parens.extract(self.tree);
                self.infer_type_expr_maybe(data.value)
            },

            syntax::Expression::Index(index) => {
                let data = index.extract(self.tree);
                let target_type = self.infer_type_expr_maybe(data.target);
                let index_type = self.infer_type_expr_maybe(data.index);

                let target_type = target_type?;
                let element = match &target_type {
                    Type::Array(Some(element), _) => Some(element),
                    Type::Ptr(_, Some(inner), _) => match &**inner {
                        Type::Array(Some(element), _) => Some(element),
                        _ => None,
                    },
                    _ => None,
                };

                let Some(element) = element else {
                    self.errors.borrow_mut().push(Error::InvalidIndexTarget(index, target_type));
                    return None;
                };

                if let Some(typ) = index_type {
                    if !matches!(typ, Type::Scalar(TypeScalar::I32 | TypeScalar::U32)) {
                        self.errors.borrow_mut().push(Error::InvalidIndexIndex(index, typ));
                    }
                }

                Some(Type::clone(element))
            },

            syntax::Expression::Member(expr_member) => {
                let data = expr_member.extract(self.tree);
                let target_type = self.infer_type_expr_maybe(data.target)?;

                let member = data.member?;
                let member_text = &self.source[member.byte_range()];

                match target_type {
                    Type::Vec(count, scalar) => {
                        let xyzw = &b"xyzw"[..count as usize];
                        let rgba = &b"rgba"[..count as usize];

                        let is_xyzw = member_text.bytes().all(|x| xyzw.contains(&x));
                        let is_rgba = member_text.bytes().all(|x| rgba.contains(&x));
                        if !is_xyzw && !is_rgba {
                            self.errors
                                .borrow_mut()
                                .push(Error::InvalidMember(expr_member, target_type));
                        }

                        if count == 1 {
                            scalar.map(Type::Scalar)
                        } else {
                            Some(Type::Vec(count, scalar))
                        }
                    },
                    Type::Struct(strukt) => {
                        let data = strukt.extract(self.tree);
                        let fields = data.fields?;
                        for field in fields.fields(self.tree) {
                            let data_field = field.extract(self.tree);
                            if let Some(field_name) = data_field.name {
                                let field_text = &self.source[field_name.byte_range()];
                                if field_text == member_text {
                                    let spec = data_field.typ?;
                                    self.references.borrow_mut().insert(
                                        member.index(),
                                        Reference::User(ReferenceNode::StructField(field)),
                                    );
                                    return self.type_from_specifier(spec, self.tree, self.source);
                                }
                            }
                        }

                        self.errors
                            .borrow_mut()
                            .push(Error::InvalidMember(expr_member, target_type));

                        None
                    },
                    _ => {
                        self.errors
                            .borrow_mut()
                            .push(Error::InvalidMember(expr_member, target_type));
                        None
                    },
                }
            },
        }
    }

    fn type_of_reference(&self, reference: &Reference) -> Option<Type> {
        match reference {
            Reference::User(node) => match node {
                ReferenceNode::Struct(strukt) => Some(Type::Type(Rc::new(Type::Struct(*strukt)))),
                ReferenceNode::Fn(func) => Some(Type::Fn(*func)),

                ReferenceNode::Alias(alias) => {
                    let spec = alias.extract(self.tree).typ?;
                    self.type_from_specifier(spec, self.tree, self.source)
                },

                ReferenceNode::StructField(field) => {
                    let data = field.extract(self.tree);
                    self.type_of_variable(data.typ, None)
                },
                ReferenceNode::Const(konst) => {
                    let data = konst.extract(self.tree);
                    self.type_of_variable(data.typ, data.value)
                },
                ReferenceNode::Override(overide) => {
                    let data = overide.extract(self.tree);
                    self.type_of_variable(data.typ, data.value)
                },
                ReferenceNode::Var(war) => {
                    let data = war.extract(self.tree);
                    self.type_of_variable(data.typ, data.value)
                },
                ReferenceNode::FnParameter(param) => {
                    let data = param.extract(self.tree);
                    self.type_of_variable(data.typ, None)
                },
                ReferenceNode::Let(lett) => {
                    let data = lett.extract(self.tree);
                    self.type_of_variable(data.typ, data.value)
                },
            },
            Reference::BuiltinFunction(_) => None,
            Reference::BuiltinTypeAlias(_, original) => self.parse_type_specifier(original),
            Reference::Type(typ) => Some(Type::Type(Rc::new(typ.clone()))),
        }
    }

    fn type_of_variable(
        &self,
        spec: Option<syntax::TypeSpecifier>,
        value: Option<syntax::Expression>,
    ) -> Option<Type> {
        if let Some(spec) = spec {
            self.type_from_specifier(spec, self.tree, self.source)
        } else {
            self.infer_type_expr(value?)
        }
    }

    fn parse_type_specifier(&self, text: &str) -> Option<Type> {
        let mut parser = parse::Parser::new();
        let output = parse::parse_type_specifier(&mut parser, text);
        let tree = &output.tree;
        let node = syntax::SyntaxNode::new(tree.root(), tree.root_index());
        let spec = syntax::TypeSpecifier::new(node)?;
        self.type_from_specifier(spec, tree, text)
    }

    fn type_from_specifier(
        &self,
        spec: syntax::TypeSpecifier,
        tree: &parse::Tree,
        source: &str,
    ) -> Option<Type> {
        let (identifier, templates) = match spec {
            syntax::TypeSpecifier::Identifier(name) => (name, None),
            syntax::TypeSpecifier::IdentifierWithTemplate(identifier) => {
                let data = identifier.extract(tree);
                (data.name?, data.templates)
            },
        };

        let mut params = templates
            .into_iter()
            .flat_map(|x| x.parameters(tree).filter_map(|x| x.extract(tree).value));

        let name = &source[identifier.byte_range()];

        let type_inner = |context: &Self, parameter: Option<syntax::Expression>| {
            let spec = match parameter? {
                syntax::Expression::Identifier(x) => syntax::TypeSpecifier::Identifier(x),
                syntax::Expression::IdentifierWithTemplate(x) => {
                    syntax::TypeSpecifier::IdentifierWithTemplate(x)
                },
                _ => return None,
            };
            context.type_from_specifier(spec, tree, source)
        };

        let scalar = |context: &Self, parameter: Option<syntax::Expression>| match type_inner(
            context, parameter,
        ) {
            Some(Type::Scalar(scalar)) => Some(scalar),
            _ => None,
        };

        let integer = |_: &Self, parameter: Option<syntax::Expression>| match parameter? {
            syntax::Expression::IntegerDecimal(integer) => source[integer.byte_range()]
                .trim_end_matches(|x| x == 'i' || x == 'u')
                .parse::<u32>()
                .ok(),
            syntax::Expression::IntegerHex(integer) => {
                let digits = source[integer.byte_range()]
                    .trim_start_matches("0x")
                    .trim_start_matches("0X")
                    .trim_end_matches(|x| x == 'i' || x == 'u');
                u32::from_str_radix(digits, 16).ok()
            },
            _ => None,
        };

        let address_space = |_: &Self, parameter: Option<syntax::Expression>| match parameter? {
            syntax::Expression::Identifier(name) => {
                AddressSpace::from_str(&source[name.byte_range()])
            },
            _ => None,
        };

        let format = |_: &Self, parameter: Option<syntax::Expression>| match parameter? {
            syntax::Expression::Identifier(name) => {
                TextureFormat::from_str(&source[name.byte_range()])
            },
            _ => None,
        };

        let access = |_: &Self, parameter: Option<syntax::Expression>| match parameter? {
            syntax::Expression::Identifier(name) => {
                AccessMode::from_str(&source[name.byte_range()])
            },
            _ => None,
        };

        if let Some(reference) = self.references.borrow().get(&identifier.index()) {
            if let Some(typ) = self.type_of_reference(reference) {
                return Some(typ);
            }
        }

        match name {
            "i32" => Some(Type::Scalar(TypeScalar::I32)),
            "u32" => Some(Type::Scalar(TypeScalar::U32)),
            "f32" => Some(Type::Scalar(TypeScalar::F32)),
            "f16" => Some(Type::Scalar(TypeScalar::F16)),

            "bool" => Some(Type::Scalar(TypeScalar::Bool)),

            "vec2" => Some(Type::Vec(2, scalar(self, params.next()))),
            "vec3" => Some(Type::Vec(3, scalar(self, params.next()))),
            "vec4" => Some(Type::Vec(4, scalar(self, params.next()))),

            "mat2x2" => Some(Type::Mat(2, 2, scalar(self, params.next()))),
            "mat2x3" => Some(Type::Mat(2, 3, scalar(self, params.next()))),
            "mat2x4" => Some(Type::Mat(2, 4, scalar(self, params.next()))),

            "mat3x2" => Some(Type::Mat(3, 2, scalar(self, params.next()))),
            "mat3x3" => Some(Type::Mat(3, 3, scalar(self, params.next()))),
            "mat3x4" => Some(Type::Mat(3, 4, scalar(self, params.next()))),

            "mat4x2" => Some(Type::Mat(4, 2, scalar(self, params.next()))),
            "mat4x3" => Some(Type::Mat(4, 3, scalar(self, params.next()))),
            "mat4x4" => Some(Type::Mat(4, 4, scalar(self, params.next()))),

            "ptr" => Some(Type::Ptr(
                address_space(self, params.next()),
                type_inner(self, params.next()).map(Rc::new),
                access(self, params.next()),
            )),

            "array" => Some(Type::Array(
                type_inner(self, params.next()).map(Rc::new),
                integer(self, params.next()).and_then(NonZeroU32::new),
            )),

            "atomic" => Some(Type::Atomic(scalar(self, params.next()))),

            "sampler" => Some(Type::Sampler),
            "sampler_comparison" => Some(Type::SamplerComparison),
            "texture_depth_2d" => Some(Type::TextureDepth2d),
            "texture_depth_2d_array" => Some(Type::TextureDepth2dArray),
            "texture_depth_cube" => Some(Type::TextureDepthCube),
            "texture_depth_cube_array" => Some(Type::TextureDepthCubeArray),
            "texture_depth_multisampled_2d" => Some(Type::TextureDepthMultisampled2d),
            "texture_external" => Some(Type::TextureExternal),

            "texture_1d" => Some(Type::Texture1d(scalar(self, params.next()))),
            "texture_2d" => Some(Type::Texture2d(scalar(self, params.next()))),
            "texture_2d_array" => Some(Type::Texture2dArray(scalar(self, params.next()))),
            "texture_3d" => Some(Type::Texture3d(scalar(self, params.next()))),
            "texture_cube" => Some(Type::TextureCube(scalar(self, params.next()))),
            "texture_cube_array" => Some(Type::TextureCubeArray(scalar(self, params.next()))),
            "texture_multisampled_2d" => {
                Some(Type::TextureMultisampled2d(scalar(self, params.next())))
            },

            "texture_storage_1d" => Some(Type::TextureStorage1d(
                format(self, params.next()),
                access(self, params.next()),
            )),
            "texture_storage_2d" => Some(Type::TextureStorage2d(
                format(self, params.next()),
                access(self, params.next()),
            )),
            "texture_storage_2d_array" => Some(Type::TextureStorage2dArray(
                format(self, params.next()),
                access(self, params.next()),
            )),
            "texture_storage_3d" => Some(Type::TextureStorage3d(
                format(self, params.next()),
                access(self, params.next()),
            )),

            // TODO: add samplers, textures, etc.
            _ => None,
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
    node: ReferenceNode,
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

    pub fn insert(&mut self, name: &'a str, node: ReferenceNode) {
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

    pub fn get(&mut self, name: &'a str) -> Option<ReferenceNode> {
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

    fn get_slot(slot: &mut ScopeSymbol, active_scopes: &[ScopeId]) -> Option<ReferenceNode> {
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

    fn iter_symbols(&mut self) -> impl Iterator<Item = (&'a str, ReferenceNode)> + '_ {
        self.symbols.iter_mut().filter_map(|(name, slot)| {
            let node = Self::get_slot(slot, &self.active_scopes)?;
            Some((*name, node))
        })
    }
}

#[derive(Debug)]
pub struct ResolvedSymbol {
    pub name: String,
    pub reference: Reference,
    pub typ: Option<Type>,
}

#[derive(Debug, Clone)]
pub enum Type {
    /// The type of a specific type. For example, `u32` in source code would have the type
    /// `Type(u32)`, while the number `123u` would have the actual type `u32`.
    Type(Rc<Type>),

    Scalar(TypeScalar),
    Vec(u8, Option<TypeScalar>),
    Mat(u8, u8, Option<TypeScalar>),
    Atomic(Option<TypeScalar>),
    Array(Option<Rc<Type>>, Option<NonZeroU32>),
    Struct(syntax::DeclStruct),
    Fn(syntax::DeclFn),

    Ptr(Option<AddressSpace>, Option<Rc<Type>>, Option<AccessMode>),

    Sampler,
    SamplerComparison,
    TextureDepth2d,
    TextureDepth2dArray,
    TextureDepthCube,
    TextureDepthCubeArray,
    TextureDepthMultisampled2d,
    TextureExternal,

    Texture1d(Option<TypeScalar>),
    Texture2d(Option<TypeScalar>),
    Texture2dArray(Option<TypeScalar>),
    Texture3d(Option<TypeScalar>),
    TextureCube(Option<TypeScalar>),
    TextureCubeArray(Option<TypeScalar>),
    TextureMultisampled2d(Option<TypeScalar>),

    TextureStorage1d(Option<TextureFormat>, Option<AccessMode>),
    TextureStorage2d(Option<TextureFormat>, Option<AccessMode>),
    TextureStorage2dArray(Option<TextureFormat>, Option<AccessMode>),
    TextureStorage3d(Option<TextureFormat>, Option<AccessMode>),
}

impl Type {
    pub fn fmt<F: std::fmt::Write>(
        &self,
        f: &mut F,
        tree: &parse::Tree,
        source: &str,
    ) -> std::fmt::Result {
        struct Maybe<'a, T>(&'a Option<T>);

        impl<T: std::fmt::Display> std::fmt::Display for Maybe<'_, T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.0 {
                    Some(inner) => write!(f, "{inner}"),
                    None => write!(f, "?"),
                }
            }
        }

        let type_maybe = |f: &mut F, typ: &Option<Rc<Type>>| -> std::fmt::Result {
            if let Some(typ) = typ {
                Type::fmt(typ, f, tree, source)
            } else {
                write!(f, "?")
            }
        };

        match self {
            Type::Type(inner) => {
                write!(f, "type<")?;
                inner.fmt(f, tree, source)?;
                write!(f, ">")?;
                Ok(())
            },

            Type::Scalar(scalar) => write!(f, "{scalar}"),

            Type::Vec(dims, scalar) => write!(f, "vec{dims}<{}>", Maybe(scalar)),

            Type::Mat(cols, rows, scalar) => write!(f, "mat{cols}x{rows}<{}>", Maybe(scalar)),

            Type::Atomic(scalar) => write!(f, "atomic<{}>", Maybe(scalar)),

            Type::Ptr(address, inner, access) => {
                write!(f, "ptr<{}, ", Maybe(address))?;
                type_maybe(f, inner)?;
                if let Some(access) = access {
                    write!(f, ", {access}")?;
                }
                write!(f, ">")?;
                Ok(())
            },

            Type::Array(inner, count) => {
                write!(f, "array<")?;
                type_maybe(f, inner)?;
                if let Some(count) = count {
                    write!(f, ", {count}")?;
                }
                write!(f, ">")?;
                Ok(())
            },

            Type::Struct(strukt) => {
                let Some(name) = strukt.extract(tree).name else { return write!(f, "?struct?") };
                write!(f, "{}", &source[name.byte_range()])
            },
            Type::Fn(func) => {
                let Some(name) = func.extract(tree).name else { return write!(f, "?fn?") };
                write!(f, "{}", &source[name.byte_range()])
            },

            Type::Sampler => write!(f, "sampler"),
            Type::SamplerComparison => write!(f, "sampler_comparison"),
            Type::TextureDepth2d => write!(f, "texture_depth_2d"),
            Type::TextureDepth2dArray => write!(f, "texture_depth_2d_array"),
            Type::TextureDepthCube => write!(f, "texture_depth_cube"),
            Type::TextureDepthCubeArray => write!(f, "texture_depth_cube_array"),
            Type::TextureDepthMultisampled2d => write!(f, "texture_depth_multisampled_2d"),
            Type::TextureExternal => write!(f, "texture_external"),

            Type::Texture1d(scalar) => write!(f, "texture_1d<{}>", Maybe(scalar)),
            Type::Texture2d(scalar) => write!(f, "texture_2d<{}>", Maybe(scalar)),
            Type::Texture2dArray(scalar) => write!(f, "texture_2d_array<{}>", Maybe(scalar)),
            Type::Texture3d(scalar) => write!(f, "texture_3d<{}>", Maybe(scalar)),
            Type::TextureCube(scalar) => write!(f, "texture_cube<{}>", Maybe(scalar)),
            Type::TextureCubeArray(scalar) => write!(f, "texture_cube_array<{}>", Maybe(scalar)),
            Type::TextureMultisampled2d(scalar) => {
                write!(f, "texture_multisampled_2d<{}>", Maybe(scalar))
            },

            Type::TextureStorage1d(format, access) => {
                write!(f, "texture_storage_1d<{}, {}>", Maybe(format), Maybe(access))
            },
            Type::TextureStorage2d(format, access) => {
                write!(f, "texture_storage_2d<{}, {}>", Maybe(format), Maybe(access))
            },
            Type::TextureStorage2dArray(format, access) => {
                write!(f, "texture_storage_2d_array<{}, {}>", Maybe(format), Maybe(access))
            },
            Type::TextureStorage3d(format, access) => {
                write!(f, "texture_storage_3d<{}, {}>", Maybe(format), Maybe(access))
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone, Copy)]
pub enum AddressSpace {
    Function,
    Private,
    Workgroup,
    Uniform,
    Storage,
    Handle,
}

impl AddressSpace {
    pub fn from_str(text: &str) -> Option<AddressSpace> {
        Some(match text {
            "function" => AddressSpace::Function,
            "private" => AddressSpace::Private,
            "workgroup" => AddressSpace::Workgroup,
            "uniform" => AddressSpace::Uniform,
            "storage" => AddressSpace::Storage,
            "handle" => AddressSpace::Handle,
            _ => return None,
        })
    }

    fn as_str(self) -> &'static str {
        match self {
            AddressSpace::Function => "function",
            AddressSpace::Private => "private",
            AddressSpace::Workgroup => "workgroup",
            AddressSpace::Uniform => "uniform",
            AddressSpace::Storage => "storage",
            AddressSpace::Handle => "handle",
        }
    }
}

impl std::fmt::Display for AddressSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_str().fmt(f)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TextureFormat {
    Rgba8Unorm,
    Rgba8Snorm,
    Rgba8Uint,
    Rgba8Sint,
    Rgba16Uint,
    Rgba16Sint,
    Rgba16Float,
    R32Uint,
    R32Sint,
    R32Float,
    Rg32Uint,
    Rg32Sint,
    Rg32Float,
    Rgba32Uint,
    Rgba32Sint,
    Rgba32Float,
    Bgra8Unorm,
}

impl TextureFormat {
    pub fn from_str(text: &str) -> Option<TextureFormat> {
        Some(match text {
            "rgba8unorm" => TextureFormat::Rgba8Unorm,
            "rgba8snorm" => TextureFormat::Rgba8Snorm,
            "rgba8uint" => TextureFormat::Rgba8Uint,
            "rgba8sint" => TextureFormat::Rgba8Sint,
            "rgba16uint" => TextureFormat::Rgba16Uint,
            "rgba16sint" => TextureFormat::Rgba16Sint,
            "rgba16float" => TextureFormat::Rgba16Float,
            "r32uint" => TextureFormat::R32Uint,
            "r32sint" => TextureFormat::R32Sint,
            "r32float" => TextureFormat::R32Float,
            "rg32uint" => TextureFormat::Rg32Uint,
            "rg32sint" => TextureFormat::Rg32Sint,
            "rg32float" => TextureFormat::Rg32Float,
            "rgba32uint" => TextureFormat::Rgba32Uint,
            "rgba32sint" => TextureFormat::Rgba32Sint,
            "rgba32float" => TextureFormat::Rgba32Float,
            "bgra8unorm" => TextureFormat::Bgra8Unorm,
            _ => return None,
        })
    }

    fn as_str(self) -> &'static str {
        match self {
            TextureFormat::Rgba8Unorm => "rgba8unorm",
            TextureFormat::Rgba8Snorm => "rgba8snorm",
            TextureFormat::Rgba8Uint => "rgba8uint",
            TextureFormat::Rgba8Sint => "rgba8sint",
            TextureFormat::Rgba16Uint => "rgba16uint",
            TextureFormat::Rgba16Sint => "rgba16sint",
            TextureFormat::Rgba16Float => "rgba16float",
            TextureFormat::R32Uint => "r32uint",
            TextureFormat::R32Sint => "r32sint",
            TextureFormat::R32Float => "r32float",
            TextureFormat::Rg32Uint => "rg32uint",
            TextureFormat::Rg32Sint => "rg32sint",
            TextureFormat::Rg32Float => "rg32float",
            TextureFormat::Rgba32Uint => "rgba32uint",
            TextureFormat::Rgba32Sint => "rgba32sint",
            TextureFormat::Rgba32Float => "rgba32float",
            TextureFormat::Bgra8Unorm => "bgra8unorm",
        }
    }
}

impl std::fmt::Display for TextureFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_str().fmt(f)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AccessMode {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

impl AccessMode {
    pub fn from_str(text: &str) -> Option<AccessMode> {
        Some(match text {
            "read_only" => AccessMode::ReadOnly,
            "write_only" => AccessMode::WriteOnly,
            "read_write" => AccessMode::ReadWrite,
            _ => return None,
        })
    }

    fn as_str(self) -> &'static str {
        match self {
            AccessMode::ReadOnly => "read_only",
            AccessMode::WriteOnly => "write_only",
            AccessMode::ReadWrite => "read_write",
        }
    }
}

impl std::fmt::Display for AccessMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_str().fmt(f)
    }
}
