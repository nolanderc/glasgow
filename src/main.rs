mod analyze;
mod parse;
mod syntax;
mod test_util;
mod util;
mod workspace;

use std::{
    collections::{BTreeMap, VecDeque},
    io::{Read, Write as _},
    rc::Rc,
};

use anyhow::{Context as _, Result};
use lsp::notification::Notification as _;

#[derive(Debug, clap::Parser)]
struct Arguments {
    #[clap(subcommand)]
    communication: Option<Communication>,
}

#[derive(Debug, clap::Subcommand)]
enum Communication {
    /// Communicate over stdin and stdout.
    Stdio,

    /// Communicate over a TCP socket.
    Socket(CommunicationSocket),

    /// Proxy stdin and stdout to a language server listening for a TCP socket.
    ///
    /// This is intended to be used during development as it means we can both get output logs in
    /// another terminal from the editor, and recompile and run a new version of the LSP easier.
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
    tracing_subscriber::fmt().init();

    let (connection, threads) = match &arguments.communication {
        None | Some(Communication::Stdio) => lsp_server::Connection::stdio(),
        Some(Communication::Socket(socket)) => {
            tracing::info!(%socket.addr, socket.port, "waiting on connection");
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

    router.handle_request::<lsp::request::HoverRequest>(|state, request| {
        let location = &request.text_document_position_params;
        let document = state.workspace.document_mut(&location.text_document.uri)?;
        let offset = document
            .offset_utf8_from_position_utf16(location.position)
            .context("invalid position")?;

        let Some((symbol, token)) = document.symbol_at_offset(offset) else { return Ok(None) };

        let parsed = document.parse();
        let range = parsed.tree.node(token).byte_range();
        let markdown = symbol_documentation(symbol)?;

        Ok(Some(lsp::Hover {
            range: Some(lsp::Range {
                start: document.position_utf16_from_offset_utf8(range.start).unwrap(),
                end: document.position_utf16_from_offset_utf8(range.end).unwrap(),
            }),
            contents: lsp::HoverContents::Markup(lsp::MarkupContent {
                kind: lsp::MarkupKind::Markdown,
                value: markdown,
            }),
        }))
    });

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
}

fn publish_diagnostics_for_document(
    state: &mut State,
    document_id: lsp::VersionedTextDocumentIdentifier,
) -> Result<()> {
    let document = state.workspace.document_mut(&document_id.uri)?;
    if document.version() != Some(document_id.version) {
        tracing::info!("document out of date");
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

fn symbol_documentation(symbol: analyze::ResolvedSymbol) -> Result<String> {
    use std::fmt::Write;

    let mut markdown = String::new();

    match symbol {
        analyze::ResolvedSymbol::BuiltinFunction(function) => {
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

                writeln!(markdown, "{}", overload.signature)?;
                writeln!(markdown, "```")?;
                writeln!(markdown)?;
            }
        },
    }

    Ok(markdown)
}

fn collect_diagnostics(state: &State, uri: &lsp::Uri) -> Result<Vec<lsp::Diagnostic>> {
    let mut diagnostics = Vec::new();

    let document = state.workspace.document(uri)?;
    let parsed = document.parse();
    for error in parsed.errors.clone().iter() {
        let byte_range = error.token.byte_range();
        diagnostics.push(lsp::Diagnostic {
            range: lsp::Range {
                start: document.position_utf16_from_offset_utf8(byte_range.start).unwrap(),
                end: document.position_utf16_from_offset_utf8(byte_range.end).unwrap(),
            },
            severity: Some(lsp::DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("syntax".into()),
            message: error.message(document.content()),
            related_information: None,
            tags: None,
            data: None,
        });
    }

    Ok(diagnostics)
}

fn run_server(_arguments: Arguments, connection: lsp_server::Connection) -> Result<()> {
    tracing::info!("starting server");

    let server_capabilities = lsp::ServerCapabilities {
        hover_provider: Some(lsp::HoverProviderCapability::Simple(true)),
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
        ..Default::default()
    };

    let _initialize_params: lsp::InitializeParams =
        serde_json::from_value(connection.initialize(serde_json::to_value(server_capabilities)?)?)?;

    tracing::info!("LSP initialized");

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
        tracing::info!("invoking handler");
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

    #[test]
    fn hover() {
        let client = Client::new();

        let (source, cursors) = document_with_cursors(indoc::indoc! {r#"
            fn main() {
                $dot(vec3(1.0), vec3(1, 2, 3));
            }
        "#});

        let uri = client.open_document(&source);

        let hover_info = client.request::<lsp::request::HoverRequest>(lsp::HoverParams {
            text_document_position_params: lsp::TextDocumentPositionParams {
                text_document: lsp::TextDocumentIdentifier::new(uri.clone()),
                position: cursors[0],
            },
            work_done_progress_params: Default::default(),
        });

        assert_serialized_eq(
            hover_info,
            expect![[r#"
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
            "#]],
        );
    }
}
