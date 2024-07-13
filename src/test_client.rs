#![cfg(test)]

use color_eyre::Result;

pub struct Client {
    connection: lsp_server::Connection,
    server: Option<std::thread::JoinHandle<Result<()>>>,
    next_id: std::cell::Cell<std::num::Wrapping<i32>>,
}

impl Client {
    pub fn new() -> Client {
        let (client, server) = lsp_server::Connection::memory();
        let arguments = <crate::Arguments as clap::Parser>::parse_from([env!("CARGO_BIN_NAME")]);
        let server = std::thread::spawn(move || crate::run_server(arguments, server));
        let client = Client {
            connection: client,
            server: Some(server),
            next_id: std::cell::Cell::new(std::num::Wrapping(0)),
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

    pub fn request<R>(&self, params: R::Params) -> R::Result
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
                params: serde_json::to_value(params).unwrap(),
            }))
            .unwrap();

        let message = self.connection.receiver.recv().unwrap();
        let response = match message {
            lsp_server::Message::Response(response) => response,
            lsp_server::Message::Request(request) => {
                panic!("got unexpected request: {request:?}")
            },
            lsp_server::Message::Notification(notification) => {
                panic!("got unexpected notification: {notification:?}")
            },
        };

        assert_eq!(response.id, id);
        if let Some(error) = response.error {
            panic!("server responded with error: {:#?}", error);
        }

        let value = response.result.expect("response did not contain a value");
        serde_json::from_value(value).unwrap()
    }

    pub fn notify<N>(&self, params: N::Params)
    where
        N: lsp::notification::Notification<Params: serde::Serialize>,
    {
        self.connection
            .sender
            .send(lsp_server::Message::Notification(lsp_server::Notification {
                method: N::METHOD.into(),
                params: serde_json::to_value(params).unwrap(),
            }))
            .unwrap();
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let done = std::sync::Mutex::new(false);
        let condition = std::sync::Condvar::new();

        self.request::<lsp::request::Shutdown>(());
        self.notify::<lsp::notification::Exit>(());

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

            self.server
                .take()
                .unwrap()
                .join()
                .expect("server panicked")
                .expect("server ended with an error");

            *done.lock().unwrap() = true;
            condition.notify_all();
        })
    }
}
