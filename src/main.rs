mod analyze;
mod arena;
mod format;
mod parse;
mod syntax;
mod test_util;
mod util;
mod workspace;

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    io::{Read, Write as _},
    rc::Rc,
};

use anyhow::{anyhow, Context as _, Result};
use lsp::{notification::Notification as _, CompletionItemKind};

use crate::syntax::{Extract as _, SyntaxNodeMatch};

#[derive(Debug, clap::Parser)]
#[clap(version, author, about)]
struct Arguments {
    #[clap(subcommand)]
    communication: Option<Communication>,
}

#[derive(Debug, clap::Subcommand)]
enum Communication {
    /// Communicate over stdin and stdout.
    #[clap(name = "--stdio")]
    Stdio,

    /// Communicate over a TCP socket.
    #[clap(name = "--socket")]
    Socket(CommunicationSocket),

    /// Proxy stdin and stdout to a language server listening for a TCP socket.
    ///
    /// This is intended to be used during development as it means we can both get output logs in
    /// another terminal from the editor, and recompile and run a new version of the LSP easier.
    #[clap(name = "--proxy")]
    Proxy {
        #[clap(long, default_value = "127.0.0.1")]
        addr: std::net::IpAddr,
        #[clap(long)]
        port: u16,
    },
}

#[derive(Debug, clap::Parser)]
struct CommunicationSocket {
    #[clap(long, default_value = "0.0.0.0")]
    addr: std::net::IpAddr,
    #[clap(long)]
    port: u16,
}

fn main() -> Result<()> {
    let arguments = <Arguments as clap::Parser>::parse();
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let (connection, threads) = match &arguments.communication {
        None | Some(Communication::Stdio) => lsp_server::Connection::stdio(),
        Some(Communication::Socket(socket)) => {
            tracing::debug!(%socket.addr, socket.port, "waiting on connection");
            lsp_server::Connection::listen((socket.addr, socket.port))
                .context("could not establish connection")?
        },

        &Some(Communication::Proxy { addr, port }) => {
            run_proxy((addr, port).into())?;
            return Ok(());
        },
    };

    run_server(arguments, connection)?;
    threads.join().context("could not join IO threads")?;

    Ok(())
}

fn add_routes(router: &mut Router) {
    router.handle_notice::<lsp::notification::DidOpenTextDocument>(|state, notice| {
        let document_id = lsp::VersionedTextDocumentIdentifier {
            uri: notice.text_document.uri.clone(),
            version: notice.text_document.version,
        };
        state.workspace.create_document(notice.text_document);
        state.tasks.schedule(move |state| publish_diagnostics_for_document(state, document_id));
        Ok(())
    });

    router.handle_notice::<lsp::notification::DidCloseTextDocument>(|state, notice| {
        state.workspace.remove_document(notice.text_document);
        Ok(())
    });

    router.handle_notice::<lsp::notification::DidChangeTextDocument>(|state, notice| {
        let document = state.workspace.document_mut(&notice.text_document.uri)?;
        for change in notice.content_changes {
            document.apply_change(change).context("could not apply change to text document")?;
        }
        document.set_version(notice.text_document.version);
        document.reset();

        let document_id = notice.text_document;
        state.tasks.schedule(move |state| publish_diagnostics_for_document(state, document_id));

        Ok(())
    });

    router.handle_notice::<lsp::notification::DidSaveTextDocument>(|_state, _notice| Ok(()));

    router.handle_request::<lsp::request::DocumentDiagnosticRequest>(|state, request| {
        let diagnostics = collect_diagnostics(state, &request.text_document.uri)?;
        Ok(lsp::DocumentDiagnosticReportResult::Report(lsp::DocumentDiagnosticReport::Full(
            lsp::RelatedFullDocumentDiagnosticReport {
                related_documents: None,
                full_document_diagnostic_report: lsp::FullDocumentDiagnosticReport {
                    result_id: None,
                    items: diagnostics,
                },
            },
        )))
    });

    router.handle_request::<lsp::request::HoverRequest>(|state, request| {
        let location = &request.text_document_position_params;
        let document = state.workspace.document(&location.text_document.uri)?;
        let offset = document
            .offset_utf8_from_position_utf16(location.position)
            .context("invalid position")?;

        let Some((symbol, token)) = document.symbol_in_range(&state.workspace, offset..offset + 1)
        else {
            return Ok(None);
        };

        let parsed = document.parse();
        let range = parsed.tree.node(token).byte_range();
        let documentation = symbol_documentation(&state.workspace, &symbol, true)?;

        Ok(Some(lsp::Hover {
            range: document.range_utf16_from_range_utf8(range),
            contents: lsp::HoverContents::Markup(documentation),
        }))
    });

    router.handle_request::<lsp::request::Completion>(|state, request| {
        let location = &request.text_document_position;
        let document = state.workspace.document(&location.text_document.uri)?;
        let offset = document
            .offset_utf8_from_position_utf16(location.position)
            .context("invalid position")?;

        let Some((symbols, _token)) =
            document.visible_symbols_in_range(&state.workspace, 0..offset)
        else {
            return Ok(None);
        };

        let mut completions = Vec::with_capacity(symbols.len());

        for symbol in symbols {
            let documentation = symbol_documentation(&state.workspace, &symbol, false)?;
            let mut item = lsp::CompletionItem {
                label: symbol.name,
                documentation: Some(lsp::Documentation::MarkupContent(documentation)),
                label_details: symbol.typ.as_ref().map(|typ| lsp::CompletionItemLabelDetails {
                    detail: Some(format!(": {}", state.workspace.format_type(typ))),
                    ..Default::default()
                }),
                ..Default::default()
            };

            item.kind = Some(match &symbol.reference {
                analyze::Reference::User(node) => match node {
                    analyze::ReferenceNode::Alias(_) => lsp::CompletionItemKind::CLASS,
                    analyze::ReferenceNode::Struct(_) => lsp::CompletionItemKind::STRUCT,
                    analyze::ReferenceNode::StructField(_) => lsp::CompletionItemKind::FIELD,

                    analyze::ReferenceNode::Fn(_) => lsp::CompletionItemKind::FUNCTION,

                    analyze::ReferenceNode::Const(_) | analyze::ReferenceNode::Override(_) => {
                        lsp::CompletionItemKind::CONSTANT
                    },

                    analyze::ReferenceNode::ConstAssert(_) => lsp::CompletionItemKind::KEYWORD,

                    analyze::ReferenceNode::Var(_)
                    | analyze::ReferenceNode::FnParameter(_)
                    | analyze::ReferenceNode::Let(_) => lsp::CompletionItemKind::VARIABLE,
                },
                analyze::Reference::BuiltinFunction(_) => CompletionItemKind::FUNCTION,
                analyze::Reference::BuiltinTypeAlias(_, _) => CompletionItemKind::CLASS,
                analyze::Reference::BuiltinType(_) => CompletionItemKind::CLASS,
                analyze::Reference::Swizzle(_, _) => CompletionItemKind::FIELD,
                analyze::Reference::AccessMode(_)
                | analyze::Reference::AddressSpace(_)
                | analyze::Reference::TextureFormat(_)
                | analyze::Reference::Attribute(_, _)
                | analyze::Reference::AttributeBuiltin(_, _) => CompletionItemKind::KEYWORD,
            });

            completions.push(item);
        }

        Ok(Some(lsp::CompletionResponse::Array(completions)))
    });

    router.handle_request::<lsp::request::GotoDefinition>(|state, request| {
        let location = request.text_document_position_params;
        let source_document = state.workspace.document(&location.text_document.uri)?;
        let offset = source_document
            .offset_utf8_from_position_utf16(location.position)
            .context("invalid position")?;

        let Some((symbol, _token)) =
            source_document.symbol_in_range(&state.workspace, offset..offset + 1)
        else {
            return Ok(None);
        };

        let Some(name) = symbol.reference.name(&state.workspace) else { return Ok(None) };

        let range = name.syntax.byte_range();

        Ok(Some(lsp::GotoDefinitionResponse::Scalar(lsp::Location {
            uri: location.text_document.uri,
            range: source_document.range_utf16_from_range_utf8(range).unwrap(),
        })))
    });

    router.handle_request::<lsp::request::References>(|state, request| {
        let location = request.text_document_position;
        let document = state.workspace.document(&location.text_document.uri)?;
        let offset = document
            .offset_utf8_from_position_utf16(location.position)
            .context("invalid position")?;

        let Some((symbol, _token)) = document.symbol_in_range(&state.workspace, offset..offset + 1)
        else {
            return Ok(None);
        };

        let parsed = document.parse();
        let references = document.find_all_references(&state.workspace, &symbol.reference);

        let mut locations = Vec::with_capacity(references.len());
        for reference in references {
            let Some(range) = parsed.tree.byte_range_total(reference) else { continue };
            locations.push(lsp::Location {
                uri: location.text_document.uri.clone(),
                range: document.range_utf16_from_range_utf8(range).unwrap(),
            });
        }

        Ok(Some(locations))
    });

    router.handle_request::<lsp::request::Rename>(|state, request| {
        let location = request.text_document_position;
        let document = state.workspace.document(&location.text_document.uri)?;
        let offset = document
            .offset_utf8_from_position_utf16(location.position)
            .context("invalid position")?;

        let Some((symbol, _token)) = document.symbol_in_range(&state.workspace, offset..offset + 1)
        else {
            return Ok(None);
        };

        match symbol.reference {
            analyze::Reference::User(_) => {},
            _ => return Err(anyhow!("cannot rename builtin functions/types")),
        }

        let mut document_edits = Vec::new();

        let references = document.find_all_references(&state.workspace, &symbol.reference);
        let parsed = document.parse();

        for reference in references {
            let Some(identifier) = <syntax::Token!(Identifier)>::from_tree(&parsed.tree, reference)
            else {
                continue;
            };

            document_edits.push(lsp::TextEdit {
                range: document.range_utf16_from_range_utf8(identifier.byte_range()).unwrap(),
                new_text: request.new_name.clone(),
            });
        }

        // apply edits starting from the end of the file to the beginning.
        // this way we don't have to recompute ranges as earlier ranges are unaffected by changes
        // made further down the file.
        document_edits.sort_by(|a, b| b.range.start.cmp(&a.range.start));

        #[allow(clippy::mutable_key_type)]
        let mut changes = HashMap::new();
        changes.insert(location.text_document.uri, document_edits);

        Ok(Some(lsp::WorkspaceEdit {
            changes: Some(changes),
            document_changes: None,
            change_annotations: None,
        }))
    });

    router.handle_request::<lsp::request::Formatting>(|state, request| {
        let document = state.workspace.document(&request.text_document.uri)?;
        let parsed = document.parse();
        let formatted = format::format(&parsed.tree, document.content());
        Ok(Some(vec![lsp::TextEdit {
            range: document.range_utf16_from_range_utf8(0..document.content().len()).unwrap(),
            new_text: formatted,
        }]))
    });
}

fn publish_diagnostics_for_document(
    state: &mut State,
    document_id: lsp::VersionedTextDocumentIdentifier,
) -> Result<()> {
    let document = state.workspace.document(&document_id.uri)?;
    if document.version() != Some(document_id.version) {
        tracing::debug!("document out of date");
        // document is out-of-date
        return Ok(());
    }

    let diagnostics = collect_diagnostics(state, &document_id.uri)?;
    state.notify::<lsp::notification::PublishDiagnostics>(lsp::PublishDiagnosticsParams {
        uri: document_id.uri,
        diagnostics,
        version: Some(document_id.version),
    })?;

    Ok(())
}

fn collect_diagnostics(state: &State, uri: &lsp::Uri) -> Result<Vec<lsp::Diagnostic>> {
    let mut diagnostics = Vec::new();

    let document = state.workspace.document(uri)?;

    let parsed = document.parse();
    for error in parsed.errors.clone().iter() {
        diagnostics.push(lsp::Diagnostic {
            range: document.range_utf16_from_range_utf8(error.token.byte_range()).unwrap(),
            severity: Some(lsp::DiagnosticSeverity::ERROR),
            source: Some("syntax".into()),
            message: error.message(document.content()),
            ..Default::default()
        });
    }

    let global_scope = document.global_scope(&state.workspace);
    for (name, duplicate) in global_scope.errors.iter() {
        for conflict in duplicate.conflicts.iter() {
            let Some(ident) = conflict.name_in_tree(&parsed.tree) else { continue };
            diagnostics.push(lsp::Diagnostic {
                range: document.range_utf16_from_range_utf8(ident.byte_range()).unwrap(),
                severity: Some(lsp::DiagnosticSeverity::ERROR),
                message: format!("multiple declarations with the name `{name}`"),
                ..Default::default()
            })
        }
    }

    let mut context = crate::analyze::DocumentContext::new(&state.workspace, document);

    for decl in syntax::root(&parsed.tree).decls(&parsed.tree) {
        let errors = std::mem::take(context.analyze_decl(decl));
        for error in errors {
            match &error {
                analyze::Error::UnresolvedReference(unresolved) => {
                    let range = unresolved.node.byte_range();
                    let text = &document.content()[range.clone()];
                    diagnostics.push(lsp::Diagnostic {
                        range: document.range_utf16_from_range_utf8(range).unwrap(),
                        severity: Some(lsp::DiagnosticSeverity::ERROR),
                        message: format!("unresolved reference to `{text}`"),
                        ..Default::default()
                    });
                },

                analyze::Error::InvalidCallTarget(expr, typ) => {
                    let data = expr.extract(&parsed.tree);
                    let index = data.target.map(|x| x.index()).unwrap_or(expr.index());
                    let Some(range) = parsed.tree.byte_range_total(index) else { continue };
                    diagnostics.push(lsp::Diagnostic {
                        range: document.range_utf16_from_range_utf8(range).unwrap(),
                        severity: Some(lsp::DiagnosticSeverity::ERROR),
                        message: format!(
                            "cannot call value of type `{}`",
                            state.workspace.format_type(typ)
                        ),
                        ..Default::default()
                    });
                },
                analyze::Error::InvalidIndexTarget(expr, typ) => {
                    let data = expr.extract(&parsed.tree);
                    let index = data.target.map(|x| x.index()).unwrap_or(expr.index());
                    let Some(range) = parsed.tree.byte_range_total(index) else { continue };
                    diagnostics.push(lsp::Diagnostic {
                        range: document.range_utf16_from_range_utf8(range).unwrap(),
                        severity: Some(lsp::DiagnosticSeverity::ERROR),
                        message: format!(
                            "cannot index into value of type `{}`",
                            state.workspace.format_type(typ)
                        ),
                        ..Default::default()
                    });
                },
                analyze::Error::InvalidIndexIndex(expr, typ) => {
                    let data = expr.extract(&parsed.tree);
                    let index = data.index.map(|x| x.index()).unwrap_or(expr.index());
                    let Some(range) = parsed.tree.byte_range_total(index) else { continue };
                    diagnostics.push(lsp::Diagnostic {
                        range: document.range_utf16_from_range_utf8(range).unwrap(),
                        severity: Some(lsp::DiagnosticSeverity::ERROR),
                        message: format!(
                            "invalid index type `{}`",
                            state.workspace.format_type(typ)
                        ),
                        ..Default::default()
                    });
                },
                analyze::Error::InvalidMember(expr, typ) => {
                    let data = expr.extract(&parsed.tree);
                    let index = data.member.map(|x| x.index()).unwrap_or(expr.index());
                    let Some(range) = parsed.tree.byte_range_total(index) else { continue };
                    diagnostics.push(lsp::Diagnostic {
                        range: document.range_utf16_from_range_utf8(range).unwrap(),
                        severity: Some(lsp::DiagnosticSeverity::ERROR),
                        message: format!(
                            "could not find field for type `{}`",
                            state.workspace.format_type(typ)
                        ),
                        ..Default::default()
                    });
                },
                analyze::Error::InvalidOpUnary(op, typ) => {
                    let range = op.parse_node().byte_range();
                    diagnostics.push(lsp::Diagnostic {
                        range: document.range_utf16_from_range_utf8(range.clone()).unwrap(),
                        severity: Some(lsp::DiagnosticSeverity::ERROR),
                        message: format!(
                            "cannot apply operator `{}` to type `{}`",
                            &document.content()[range],
                            state.workspace.format_type(typ)
                        ),
                        ..Default::default()
                    });
                },
                analyze::Error::InvalidOpInfix(op, lhs, rhs) => {
                    let range = op.parse_node().byte_range();
                    diagnostics.push(lsp::Diagnostic {
                        range: document.range_utf16_from_range_utf8(range.clone()).unwrap(),
                        severity: Some(lsp::DiagnosticSeverity::ERROR),
                        message: format!(
                            "cannot apply operator {} to types `{}` and `{}`",
                            &document.content()[range],
                            state.workspace.format_type(lhs),
                            state.workspace.format_type(rhs),
                        ),
                        ..Default::default()
                    });
                },
                analyze::Error::InvalidCoercion(node, lhs, rhs) => {
                    let Some(range) = parsed.tree.byte_range_total(node.index()) else { continue };
                    diagnostics.push(lsp::Diagnostic {
                        range: document.range_utf16_from_range_utf8(range.clone()).unwrap(),
                        severity: Some(lsp::DiagnosticSeverity::ERROR),
                        message: format!(
                            "incompatible types `{}` and `{}`",
                            state.workspace.format_type(lhs),
                            state.workspace.format_type(rhs),
                        ),
                        ..Default::default()
                    });
                },
            }
        }
    }

    Ok(diagnostics)
}

fn symbol_documentation(
    workspace: &workspace::Workspace,
    symbol: &analyze::ResolvedSymbol,
    with_type_info: bool,
) -> Result<lsp::MarkupContent> {
    use std::fmt::Write;

    let mut markdown = String::new();

    if with_type_info {
        if let Some(typ) = &symbol.typ {
            writeln!(markdown, "```wgsl")?;
            writeln!(markdown, ": {}", workspace.format_type(typ))?;
            writeln!(markdown, "```")?;
        }
    }

    match &symbol.reference {
        &analyze::Reference::User(node) => {
            let document = workspace.document_from_id(node.document());
            let tree = &document.parse().tree;

            let snippet_range = match node {
                analyze::ReferenceNode::Fn(func) => {
                    let data = func.syntax.extract(tree);
                    let mut children = func.syntax.parse_node().children();

                    if data.body.is_some() {
                        // discard from snippet
                        _ = children.next_back();
                    }

                    tree.byte_range_total_children(children)
                },
                analyze::ReferenceNode::Alias(x) => tree.byte_range_total(x.syntax.index()),
                analyze::ReferenceNode::Const(x) => tree.byte_range_total(x.syntax.index()),
                analyze::ReferenceNode::ConstAssert(x) => tree.byte_range_total(x.syntax.index()),
                analyze::ReferenceNode::FnParameter(x) => tree.byte_range_total(x.syntax.index()),
                analyze::ReferenceNode::Let(x) => tree.byte_range_total(x.syntax.index()),
                analyze::ReferenceNode::Override(x) => tree.byte_range_total(x.syntax.index()),
                analyze::ReferenceNode::Struct(x) => tree.byte_range_total(x.syntax.index()),
                analyze::ReferenceNode::StructField(x) => tree.byte_range_total(x.syntax.index()),
                analyze::ReferenceNode::Var(x) => tree.byte_range_total(x.syntax.index()),
            };

            if let Some(range) = snippet_range {
                let text = &document.content()[range];
                writeln!(markdown, "```wgsl")?;
                for line in text.trim().lines() {
                    writeln!(markdown, "{line}")?;
                }
                writeln!(markdown, "```")?;
                writeln!(markdown)?;
            }
        },

        analyze::Reference::BuiltinFunction(function) => {
            documentation_builtin_function(&mut markdown, function)?;
        },

        analyze::Reference::BuiltinTypeAlias(name, original) => {
            writeln!(markdown, "```wgsl")?;
            writeln!(markdown, "alias {name} = {original};")?;
            writeln!(markdown, "```")?;
        },

        analyze::Reference::BuiltinType(typ) => {
            writeln!(markdown, "```wgsl")?;
            for line in typ.trim().lines() {
                writeln!(markdown, "{line}")?;
            }
            writeln!(markdown, "```")?;

            let functions = analyze::get_builtin_functions();
            if let Some(func) = functions.functions.get(*typ) {
                documentation_builtin_function(&mut markdown, func)?;
            }
        },

        analyze::Reference::Swizzle(count, scalar) => {
            writeln!(markdown, "```wgsl")?;
            if *count == 1 {
                if let Some(scalar) = scalar {
                    writeln!(markdown, "{scalar}")?;
                }
            } else {
                analyze::Type::Vec(*count, *scalar).fmt(&mut markdown, workspace)?;
                writeln!(markdown)?;
            }
            writeln!(markdown, "```")?;
        },

        analyze::Reference::AccessMode(mode) => match mode {
            analyze::AccessMode::Read => writeln!(markdown, "read-only access")?,
            analyze::AccessMode::Write => writeln!(markdown, "write-only access")?,
            analyze::AccessMode::ReadWrite => writeln!(markdown, "read and write access")?,
        },
        analyze::Reference::AddressSpace(space) => {
            match space {
                analyze::AddressSpace::Function => {
                    writeln!(
                        markdown,
                        "Address space for variables within the same function call."
                    )?;
                },
                analyze::AddressSpace::Private => {
                    writeln!(
                        markdown,
                        "Address space for variables within the same shader invocation."
                    )?;
                },
                analyze::AddressSpace::Workgroup => {
                    writeln!(
                        markdown,
                        "Address space for variables within the same compute shader workgroup."
                    )?;
                },
                analyze::AddressSpace::Uniform => {
                    writeln!(
                        markdown,
                        "Address space for uniform buffer variables within the same shader stage."
                    )?;
                },
                analyze::AddressSpace::Storage => {
                    writeln!(
                        markdown,
                        "Address space for storage buffer variables within the same shader stage."
                    )?;
                },
                analyze::AddressSpace::Handle => {
                    writeln!(
                        markdown,
                        "Address space for sampler and texture variables within the same shader stage."
                    )?;
                },
            }

            writeln!(markdown)?;
            writeln!(markdown, "Default Access Mode: `{}`", space.default_access_mode())?;
        },
        analyze::Reference::TextureFormat(format) => {
            writeln!(markdown, "Texture Format: `{}`", format)?;
        },

        analyze::Reference::Attribute(name, info) => {
            writeln!(markdown, "`@{}`", name)?;
            writeln!(markdown)?;
            writeln!(markdown, "{}", info.description)?;
            if let Some(parameters) = &info.description_parameters {
                writeln!(markdown)?;
                writeln!(markdown, "## Parameters")?;
                writeln!(markdown, "{}", parameters)?;
            }
        },

        analyze::Reference::AttributeBuiltin(name, info) => {
            writeln!(markdown, "`@builtin({}) : {}`", name, info.typ)?;

            for (stage, info_stage) in info.stages.iter().rev() {
                let direction = match info_stage.direction {
                    wgsl_spec::BuiltinValueDirection::Input => "input",
                    wgsl_spec::BuiltinValueDirection::Output => "output",
                };
                writeln!(markdown)?;
                writeln!(markdown, "## Shader Stage: `{}` ({})", stage, direction)?;
                writeln!(markdown, "{}", info_stage.description)?;
            }
        },
    }

    Ok(lsp::MarkupContent { kind: lsp::MarkupKind::Markdown, value: markdown })
}

fn documentation_builtin_function(
    markdown: &mut String,
    function: &wgsl_spec::Function,
) -> Result<(), anyhow::Error> {
    use std::fmt::Write;

    if let Some(description) = &function.description {
        writeln!(markdown, "{}", description)?;
        writeln!(markdown)?;
    }

    for overload in function.overloads.iter() {
        if let Some(description) = &overload.description {
            if !description.trim().is_empty() {
                for line in description.lines() {
                    writeln!(markdown, "{}", line)?;
                }
            }
        }

        writeln!(markdown, "```wgsl")?;
        for (typevar, values) in overload.parameterization.typevars.iter() {
            write!(markdown, "// ")?;
            match values {
                wgsl_spec::ParameterizationKind::Types(types) => {
                    write!(markdown, "`{typevar}` is ")?;
                    for (i, typ) in types.iter().enumerate() {
                        if i != 0 {
                            if i == types.len() - 1 {
                                write!(markdown, ", or ")?;
                            } else {
                                write!(markdown, ", ")?;
                            }
                        }

                        if typ.starts_with("Abstract") {
                            write!(markdown, "{typ}")?;
                        } else {
                            write!(markdown, "`{typ}`")?;
                        }
                    }
                    writeln!(markdown)?;
                },
                wgsl_spec::ParameterizationKind::Description(description) => {
                    writeln!(markdown, "`{typevar}` {description}")?;
                },
            }
        }

        for line in overload.signature.trim().lines() {
            writeln!(markdown, "{line}")?;
        }
        writeln!(markdown, "```")?;
        writeln!(markdown)?;
    }

    Ok(())
}

fn run_server(_arguments: Arguments, connection: lsp_server::Connection) -> Result<()> {
    tracing::debug!("starting server");

    let server_capabilities = lsp::ServerCapabilities {
        text_document_sync: Some(lsp::TextDocumentSyncCapability::Kind(
            lsp::TextDocumentSyncKind::FULL,
        )),
        diagnostic_provider: Some(lsp::DiagnosticServerCapabilities::Options(
            lsp::DiagnosticOptions {
                identifier: Some(env!("CARGO_BIN_NAME").into()),
                inter_file_dependencies: true,
                workspace_diagnostics: false,
                work_done_progress_options: Default::default(),
            },
        )),
        hover_provider: Some(lsp::HoverProviderCapability::Simple(true)),
        completion_provider: Some(lsp::CompletionOptions {
            resolve_provider: None,
            trigger_characters: Some(vec![".".into(), "@".into()]),
            all_commit_characters: None,
            work_done_progress_options: Default::default(),
            completion_item: None,
        }),
        definition_provider: Some(lsp::OneOf::Left(true)),
        rename_provider: Some(lsp::OneOf::Left(true)),
        references_provider: Some(lsp::OneOf::Left(true)),
        document_formatting_provider: Some(lsp::OneOf::Left(true)),
        ..Default::default()
    };

    let _initialize_params: lsp::InitializeParams =
        serde_json::from_value(connection.initialize(serde_json::to_value(server_capabilities)?)?)?;

    tracing::debug!("LSP initialized");

    let mut router = Router::default();
    add_routes(&mut router);

    let connection = Rc::new(connection);
    let mut state = State::new(connection.clone());

    loop {
        let message = match connection.receiver.try_recv() {
            Ok(message) => message,
            Err(_) => {
                if let Some(task) = state.tasks.next_task() {
                    if let Err(error) = task.run(&mut state) {
                        tracing::error!(%error, "could not complete task");
                    }
                    continue;
                }

                connection.receiver.recv()?
            },
        };

        match message {
            lsp_server::Message::Request(request) => {
                if connection.handle_shutdown(&request)? {
                    break;
                }

                let id = request.id;
                let _span =
                    tracing::error_span!("request", method=request.method, id = %id).entered();

                let mut response = lsp_server::Response { id, result: None, error: None };
                match router.dispatch(&mut state, &request.method, request.params) {
                    Ok(value) => response.result = Some(value),
                    Err(error) => {
                        tracing::error!(error = error.message);
                        response.error = Some(error);
                    },
                }

                let response = lsp_server::Message::Response(response);

                connection.sender.send(response)?;
            },
            lsp_server::Message::Response(response) => {
                tracing::warn!(?response, "got unexpected response")
            },
            lsp_server::Message::Notification(notification) => {
                if notification.method == lsp::notification::Exit::METHOD {
                    break;
                }

                let _span =
                    tracing::error_span!("notification", method = notification.method).entered();
                match router.dispatch(&mut state, &notification.method, notification.params) {
                    Ok(_) => {},
                    Err(err) => {
                        tracing::error!(err.code, err.message, "could not handle notification",)
                    },
                }
            },
        }
    }

    Ok(())
}

fn run_proxy(addr: std::net::SocketAddr) -> Result<()> {
    let socket = std::net::TcpStream::connect(addr)
        .with_context(|| format!("could not connect to peer at {addr}"))?;

    let (mut reader, mut writer) =
        (socket.try_clone().context("could not clone TCP socket")?, socket);

    let reader = std::thread::spawn(move || -> Result<()> {
        let mut stdout = std::io::stdout().lock();
        let mut buffer = [0u8; 1 << 18];
        loop {
            let count = reader.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            stdout.write_all(&buffer[..count])?;
            stdout.flush()?;
        }
        Ok(())
    });

    let writer = std::thread::spawn(move || -> Result<()> {
        let mut stdin = std::io::stdin().lock();
        let mut buffer = [0u8; 1 << 18];
        loop {
            let count = stdin.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            writer.write_all(&buffer[..count])?;
            writer.flush()?;
        }
        Ok(())
    });

    reader.join().unwrap().unwrap();
    writer.join().unwrap().unwrap();

    Ok(())
}

struct State {
    workspace: workspace::Workspace,
    tasks: TaskScheduler,
    connection: Rc<lsp_server::Connection>,
}

impl State {
    fn new(connection: Rc<lsp_server::Connection>) -> State {
        Self { workspace: Default::default(), tasks: Default::default(), connection }
    }

    fn notify<N>(&self, params: N::Params) -> Result<()>
    where
        N: lsp::notification::Notification,
    {
        self.connection.sender.send(lsp_server::Message::Notification(
            lsp_server::Notification {
                method: N::METHOD.into(),
                params: serde_json::to_value(params)?,
            },
        ))?;
        Ok(())
    }
}

#[derive(Default)]
struct TaskScheduler {
    queue: VecDeque<Task>,
}

struct Task {
    callback: TaskCallback,
}

impl Task {
    fn run(self, state: &mut State) -> Result<()> {
        (self.callback)(state)
    }
}

type TaskCallback = Box<dyn FnOnce(&mut State) -> Result<()>>;

impl TaskScheduler {
    pub fn schedule(&mut self, callback: impl FnOnce(&mut State) -> Result<()> + 'static) {
        self.queue.push_back(Task { callback: Box::new(callback) })
    }

    pub fn next_task(&mut self) -> Option<Task> {
        self.queue.pop_front()
    }
}

#[derive(Default)]
struct Router {
    handlers: BTreeMap<&'static str, RouteHandler>,
}

struct RouteHandler {
    callback: RouteCallback,
}

type RouteCallback = Box<dyn Fn(&mut State, serde_json::Value) -> RouteResult>;
type RouteResult = Result<serde_json::Value, lsp_server::ResponseError>;

impl Router {
    pub fn handle_request<R>(
        &mut self,
        callback: impl Fn(&mut State, R::Params) -> Result<R::Result> + 'static,
    ) where
        R: lsp::request::Request<
            Params: for<'de> serde::Deserialize<'de>,
            Result: serde::Serialize,
        >,
    {
        self.add_handler(
            R::METHOD,
            Box::new(move |state, params| {
                let params = match serde_json::from_value(params) {
                    Ok(params) => params,
                    Err(error) => {
                        return Err(lsp_server::ResponseError {
                            code: lsp_server::ErrorCode::InvalidParams as _,
                            message: format!("{error:#}"),
                            data: None,
                        })
                    },
                };

                let value = match callback(state, params) {
                    Ok(value) => value,
                    Err(error) => {
                        return Err(lsp_server::ResponseError {
                            code: lsp_server::ErrorCode::RequestFailed as _,
                            message: format!("{error:#}"),
                            data: None,
                        })
                    },
                };

                match serde_json::to_value(value) {
                    Ok(value) => Ok(value),
                    Err(error) => Err(lsp_server::ResponseError {
                        code: lsp_server::ErrorCode::InternalError as _,
                        message: format!("{error:#}"),
                        data: None,
                    }),
                }
            }),
        );
    }

    pub fn handle_notice<N>(
        &mut self,
        callback: impl Fn(&mut State, N::Params) -> Result<()> + 'static,
    ) where
        N: lsp::notification::Notification<Params: for<'de> serde::Deserialize<'de>>,
    {
        self.add_handler(
            N::METHOD,
            Box::new(move |state, params| {
                let params = match serde_json::from_value(params) {
                    Ok(params) => params,
                    Err(error) => {
                        return Err(lsp_server::ResponseError {
                            code: lsp_server::ErrorCode::InvalidParams as _,
                            message: format!("{error:#}"),
                            data: None,
                        })
                    },
                };

                if let Err(error) = callback(state, params) {
                    return Err(lsp_server::ResponseError {
                        code: lsp_server::ErrorCode::RequestFailed as _,
                        message: format!("{error:#}"),
                        data: None,
                    });
                }

                Ok(serde_json::Value::Null)
            }),
        );
    }

    fn add_handler(&mut self, method: &'static str, callback: RouteCallback) {
        if let Some(_previous) = self.handlers.insert(method, RouteHandler { callback }) {
            panic!("multiple handlers registered for method {method:?}");
        }
    }

    fn dispatch(&self, state: &mut State, method: &str, params: serde_json::Value) -> RouteResult {
        let handler = self.handlers.get(method).ok_or_else(|| lsp_server::ResponseError {
            code: lsp_server::ErrorCode::MethodNotFound as _,
            message: format!("unsupported method {method:?}"),
            data: None,
        })?;
        tracing::debug!("invoking handler");
        (handler.callback)(state, params)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_util::{document_with_cursors, Client};
    use expect_test::expect;

    fn assert_serialized_eq<T>(value: T, expected: expect_test::Expect)
    where
        T: serde::Serialize,
    {
        let mut found = serde_json::to_string_pretty(&value).unwrap();
        found += "\n";
        expected.assert_eq(&found);
    }

    fn check_hover(source: &str, expected: expect_test::Expect) {
        let client = Client::new();

        let (source, cursors) = document_with_cursors(source);
        let uri = client.open_document(&source);

        let mut hovers = Vec::with_capacity(cursors.len());
        for cursor in cursors {
            let hover_info = client.request::<lsp::request::HoverRequest>(lsp::HoverParams {
                text_document_position_params: lsp::TextDocumentPositionParams {
                    text_document: lsp::TextDocumentIdentifier::new(uri.clone()),
                    position: cursor,
                },
                work_done_progress_params: Default::default(),
            });
            hovers.push(hover_info);
        }

        assert_serialized_eq(hovers, expected);
    }

    fn check_completions(source: &str, expected: expect_test::Expect) {
        let client = Client::new();

        let (source, cursors) = document_with_cursors(source);
        let uri = client.open_document(&source);

        let mut completions = Vec::with_capacity(cursors.len());
        for cursor in cursors {
            let completion_info =
                client.request::<lsp::request::Completion>(lsp::CompletionParams {
                    work_done_progress_params: Default::default(),
                    text_document_position: lsp::TextDocumentPositionParams {
                        text_document: lsp::TextDocumentIdentifier::new(uri.clone()),
                        position: cursor,
                    },
                    partial_result_params: Default::default(),
                    context: None,
                });
            completions.push(completion_info);
        }

        assert_serialized_eq(completions, expected);
    }

    #[test]
    fn hover() {
        check_hover(
            indoc::indoc! {r#"
                fn main() {
                    $dot(vec3(1.0), vec3(1, 2, 3));
                }
            "#},
            expect![[r#"
                [
                  {
                    "contents": {
                      "kind": "markdown",
                      "value": "Returns the dot product of e1 and e2.\n```wgsl\n// `T` is AbstractInt, AbstractFloat, `i32`, `u32`, `f32`, or `f16`\n@const @must_use fn dot ( e1: vecN<T>, e2: vecN<T> ) -> T\n```\n\n"
                    },
                    "range": {
                      "start": {
                        "line": 1,
                        "character": 4
                      },
                      "end": {
                        "line": 1,
                        "character": 7
                      }
                    }
                  }
                ]
            "#]],
        );
    }

    #[test]
    fn hover_fields() {
        check_hover(
            indoc::indoc! {r#"
                fn main() {
                    let foo = Foo();
                    foo.$bar.$baz;
                }

                struct Foo {
                    bar: Bar,
                }

                struct Bar {
                    baz: vec3f,
                }
            "#},
            expect![[r#"
                [
                  {
                    "contents": {
                      "kind": "markdown",
                      "value": "```wgsl\n: Bar\n```\n```wgsl\nbar: Bar,\n```\n\n"
                    },
                    "range": {
                      "start": {
                        "line": 2,
                        "character": 8
                      },
                      "end": {
                        "line": 2,
                        "character": 11
                      }
                    }
                  },
                  {
                    "contents": {
                      "kind": "markdown",
                      "value": "```wgsl\n: vec3<f32>\n```\n```wgsl\nbaz: vec3f,\n```\n\n"
                    },
                    "range": {
                      "start": {
                        "line": 2,
                        "character": 12
                      },
                      "end": {
                        "line": 2,
                        "character": 15
                      }
                    }
                  }
                ]
            "#]],
        );
    }

    #[test]
    fn complete_vec_swizzle_dot() {
        check_completions(
            indoc::indoc! {r#"
                fn main() {
                    let x = vec2u(1, 2, 3);
                    x.$
                }
            "#},
            expect![[r#"
                [
                  [
                    {
                      "label": "x",
                      "labelDetails": {
                        "detail": ": u32"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nu32\n```\n"
                      }
                    },
                    {
                      "label": "y",
                      "labelDetails": {
                        "detail": ": u32"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nu32\n```\n"
                      }
                    },
                    {
                      "label": "xx",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    },
                    {
                      "label": "xy",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    },
                    {
                      "label": "yx",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    },
                    {
                      "label": "yy",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    },
                    {
                      "label": "r",
                      "labelDetails": {
                        "detail": ": u32"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nu32\n```\n"
                      }
                    },
                    {
                      "label": "g",
                      "labelDetails": {
                        "detail": ": u32"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nu32\n```\n"
                      }
                    },
                    {
                      "label": "rr",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    },
                    {
                      "label": "rg",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    },
                    {
                      "label": "gr",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    },
                    {
                      "label": "gg",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    }
                  ]
                ]
            "#]],
        );
    }

    #[test]
    fn complete_vec_swizzle_field() {
        check_completions(
            indoc::indoc! {r#"
                fn main() {
                    let x = vec2<u32>(1, 2, 3);
                    x.$
                }
            "#},
            expect![[r#"
                [
                  [
                    {
                      "label": "x",
                      "labelDetails": {
                        "detail": ": u32"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nu32\n```\n"
                      }
                    },
                    {
                      "label": "y",
                      "labelDetails": {
                        "detail": ": u32"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nu32\n```\n"
                      }
                    },
                    {
                      "label": "xx",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    },
                    {
                      "label": "xy",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    },
                    {
                      "label": "yx",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    },
                    {
                      "label": "yy",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    },
                    {
                      "label": "r",
                      "labelDetails": {
                        "detail": ": u32"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nu32\n```\n"
                      }
                    },
                    {
                      "label": "g",
                      "labelDetails": {
                        "detail": ": u32"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nu32\n```\n"
                      }
                    },
                    {
                      "label": "rr",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    },
                    {
                      "label": "rg",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    },
                    {
                      "label": "gr",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    },
                    {
                      "label": "gg",
                      "labelDetails": {
                        "detail": ": vec2<u32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nvec2<u32>\n```\n"
                      }
                    }
                  ]
                ]
            "#]],
        );
    }

    #[test]
    fn complete_struct_field() {
        check_completions(
            indoc::indoc! {r#"
                fn main() {
                    let x = Bar()
                    x.$
                }

                struct Bar {
                    foo: u32,
                    point: vec2f,
                }
            "#},
            expect![[r#"
                [
                  [
                    {
                      "label": "foo",
                      "labelDetails": {
                        "detail": ": u32"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\nfoo: u32,\n```\n\n"
                      }
                    },
                    {
                      "label": "point",
                      "labelDetails": {
                        "detail": ": vec2<f32>"
                      },
                      "kind": 5,
                      "documentation": {
                        "kind": "markdown",
                        "value": "```wgsl\npoint: vec2f,\n```\n\n"
                      }
                    }
                  ]
                ]
            "#]],
        );
    }
}
