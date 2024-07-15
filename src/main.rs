mod parse;
mod test_client;
mod util;
mod workspace;

use std::collections::BTreeMap;

use anyhow::{Context as _, Result};
use lsp::notification::Notification as _;

#[derive(Debug, clap::Parser)]
struct Arguments {
    #[clap(subcommand)]
    communication: Option<Communication>,
}

#[derive(Debug, clap::Subcommand)]
enum Communication {
    Stdio,
    Socket(CommunicationSocket),
}

#[derive(Debug, clap::Parser)]
struct CommunicationSocket {
    #[clap(default_value = "0.0.0.0")]
    addr: std::net::IpAddr,
    port: u16,
}

fn main() -> Result<()> {
    let arguments = <Arguments as clap::Parser>::parse();
    tracing_subscriber::fmt().init();

    let (connection, threads) = match &arguments.communication {
        None | Some(Communication::Stdio) => lsp_server::Connection::stdio(),
        Some(Communication::Socket(socket)) => {
            lsp_server::Connection::listen((socket.addr, socket.port))
                .context("could not establish connection")?
        },
    };

    run_server(arguments, connection)?;
    threads.join().context("could not join IO threads")?;

    Ok(())
}

fn add_routes(router: &mut Router) {
    router.handle_notice::<lsp::notification::DidOpenTextDocument>(|state, notice| {
        state.workspace.create_document(notice.text_document);
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
        Ok(())
    });

    router.handle_request::<lsp::request::HoverRequest>(|state, request| {
        let location = &request.text_document_position_params;
        let document = state.workspace.document_mut(&location.text_document.uri)?;
        let offset = document
            .offset_utf8_from_position_utf16(location.position)
            .context("invalid position")?;
        let parsed = document.parse();
        Ok(None)
    });

    router.handle_request::<lsp::request::DocumentDiagnosticRequest>(|state, request| {
        let mut diagnostics = Vec::new();

        let document = state.workspace.document_mut(&request.text_document.uri)?;
        let parsed = document.parse();
        for error in parsed.errors.clone().iter() {
            let message = match error.expected {
                parse::Expected::Token(token) => format!("expected {token:?})"),
                parse::Expected::Declaration => "expected a declaration".to_string(),
                parse::Expected::Expression => "expected an expression".to_string(),
                parse::Expected::Statement => "expected a statement".to_string(),
            };

            let byte_range = error.token.byte_range(document.content());

            diagnostics.push(lsp::Diagnostic {
                range: lsp::Range {
                    start: document.position_utf16_from_offset_utf8(byte_range.start).unwrap(),
                    end: document.position_utf16_from_offset_utf8(byte_range.end).unwrap(),
                },
                severity: Some(lsp::DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("syntax".into()),
                message,
                related_information: None,
                tags: None,
                data: None,
            });
        }

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

fn run_server(_arguments: Arguments, connection: lsp_server::Connection) -> Result<()> {
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

    let mut router = Router::default();
    add_routes(&mut router);

    let mut state = State::default();

    for message in &connection.receiver {
        match message {
            lsp_server::Message::Request(request) => {
                if connection.handle_shutdown(&request)? {
                    break;
                }

                let id = request.id;
                let _span = tracing::error_span!("request", request.method, id = %id).entered();

                let mut response = lsp_server::Response { id, result: None, error: None };
                match router.dispatch(&mut state, &request.method, request.params) {
                    Ok(value) => response.result = Some(value),
                    Err(err) => response.error = Some(err),
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

                let _span = tracing::error_span!("notification", notification.method).entered();
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

#[derive(Default)]
struct State {
    workspace: workspace::Workspace,
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
        (handler.callback)(state, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::expect;
    use test_client::Client;

    fn assert_serialized_eq<T>(value: T, expected: expect_test::Expect)
    where
        T: serde::Serialize,
    {
        let found = serde_json::to_string_pretty(&value).unwrap();
        expected.assert_eq(&found);
    }

    #[test]
    fn hover() {
        let client = Client::new();
        let uri = client.open_document("fn main() { dot(vec3(1.0), vec3(1, 2, 3)) }");

        let hover_info = client.request::<lsp::request::HoverRequest>(lsp::HoverParams {
            text_document_position_params: lsp::TextDocumentPositionParams {
                text_document: lsp::TextDocumentIdentifier::new(uri.clone()),
                position: lsp::Position { line: 0, character: 13 },
            },
            work_done_progress_params: Default::default(),
        });

        assert_serialized_eq(hover_info, expect![]);
    }
}
