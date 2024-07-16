use std::cell::Cell;

use anyhow::{anyhow, Context as _, Result};

pub struct Client {
    connection: lsp_server::Connection,
    server: Cell<Option<std::thread::JoinHandle<Result<()>>>>,
    next_id: std::cell::Cell<std::num::Wrapping<i32>>,
    ignore_notifications: bool,
}

impl Client {
    pub fn new() -> Client {
        tracing_subscriber::fmt().init();

        let (client, server) = lsp_server::Connection::memory();

        let arguments = <crate::Arguments as clap::Parser>::parse_from([env!("CARGO_BIN_NAME")]);
        let server = std::thread::Builder::new()
            .name(format!("{} - server", std::thread::current().name().unwrap_or("???")))
            .spawn(move || crate::run_server(arguments, server))
            .expect("could not spawn server thread");

        let client = Client {
            connection: client,
            server: Cell::new(Some(server)),
            next_id: std::cell::Cell::new(std::num::Wrapping(0)),
            ignore_notifications: true,
        };

        client.request::<lsp::request::Initialize>(lsp::InitializeParams::default());
        client.notify::<lsp::notification::Initialized>(lsp::InitializedParams {});

        client
    }

    fn generate_requeust_id(&self) -> lsp_server::RequestId {
        let id = self.next_id.get();
        self.next_id.set(id + std::num::Wrapping(1));
        id.0.into()
    }

    pub fn request_fallible<R>(&self, params: R::Params) -> Result<R::Result>
    where
        R: lsp::request::Request<
            Params: serde::Serialize,
            Result: for<'de> serde::Deserialize<'de>,
        >,
    {
        let id = self.generate_requeust_id();

        self.connection
            .sender
            .send(lsp_server::Message::Request(lsp_server::Request {
                id: id.clone(),
                method: R::METHOD.into(),
                params: serde_json::to_value(params)?,
            }))
            .context("could not send message to server")?;

        let response = loop {
            let message =
                self.connection.receiver.recv().context("could not receive message from server")?;

            match message {
                lsp_server::Message::Response(response) => break response,
                lsp_server::Message::Request(request) => {
                    return Err(anyhow!("got unexpected request: {request:?}"));
                },
                lsp_server::Message::Notification(notification) => {
                    if self.ignore_notifications {
                        continue;
                    }
                    return Err(anyhow!("got unexpected notification: {notification:?}"));
                },
            }
        };

        assert_eq!(response.id, id);
        if let Some(error) = response.error {
            return Err(anyhow!("server responded with error: {:#?}", error));
        }

        let value = response.result.expect("response did not contain a value");
        Ok(serde_json::from_value(value)?)
    }

    pub fn request<R>(&self, params: R::Params) -> R::Result
    where
        R: lsp::request::Request<
            Params: serde::Serialize,
            Result: for<'de> serde::Deserialize<'de>,
        >,
    {
        match self.request_fallible::<R>(params) {
            Ok(value) => value,
            Err(error) => {
                _ = self.close();
                panic!("request failed: {error:#}")
            },
        }
    }

    pub fn notify_fallible<N>(&self, params: N::Params) -> Result<()>
    where
        N: lsp::notification::Notification<Params: serde::Serialize>,
    {
        self.connection.sender.send(lsp_server::Message::Notification(
            lsp_server::Notification {
                method: N::METHOD.into(),
                params: serde_json::to_value(params)?,
            },
        ))?;
        Ok(())
    }

    pub fn notify<N>(&self, params: N::Params)
    where
        N: lsp::notification::Notification<Params: serde::Serialize>,
    {
        match self.notify_fallible::<N>(params) {
            Ok(value) => value,
            Err(error) => {
                _ = self.close();
                panic!("could not send notification: {error:#}")
            },
        }
    }

    pub fn open_document(&self, content: &str) -> lsp::Uri {
        let uri: lsp::Uri = format!("file_{}.wgsl", self.generate_requeust_id()).parse().unwrap();
        self.notify::<lsp::notification::DidOpenTextDocument>(lsp::DidOpenTextDocumentParams {
            text_document: lsp::TextDocumentItem {
                uri: uri.clone(),
                language_id: "wgsl".into(),
                version: i32::MIN,
                text: content.into(),
            },
        });
        uri
    }

    pub fn close(&self) -> Result<()> {
        let Some(server) = self.server.take() else { return Ok(()) };

        let done = std::sync::Mutex::new(false);
        let condition = std::sync::Condvar::new();

        self.request_fallible::<lsp::request::Shutdown>(())?;
        self.notify_fallible::<lsp::notification::Exit>(())?;

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);

        std::thread::scope(|scope| {
            scope.spawn(|| {
                let mut done = done.lock().unwrap();
                while !*done {
                    let Some(duration) = deadline.checked_duration_since(std::time::Instant::now())
                    else {
                        eprintln!("ERROR: server thread did not exit gracefully");
                        std::process::exit(1);
                    };
                    (done, _) =
                        condition.wait_timeout_while(done, duration, |done| !*done).unwrap();
                }
            });

            server.join().expect("server panicked").expect("server ended with an error");

            *done.lock().unwrap() = true;
            condition.notify_all();
        });

        Ok(())
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let result = self.close();
        if std::thread::panicking() {
            if let Err(error) = result {
                eprintln!("\nERROR: could not properly close client connection: {error:#}");
            }
        } else {
            result.expect("could not properly close client connection");
        }
    }
}
