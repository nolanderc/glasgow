#![cfg(test)]

mod client;

pub use client::Client;

pub fn document_with_cursors(content: &str) -> (String, Vec<lsp::Position>) {
    let mut output = String::with_capacity(content.len());
    let mut positions = Vec::new();

    let mut line = 0;
    let mut character = 0;

    for char in content.chars() {
        if char == '$' {
            positions.push(lsp::Position { line, character });
            continue;
        }

        output.push(char);

        character += char.len_utf16() as u32;
        if char == '\n' {
            line += 1;
            character = 0;
        }
    }

    (output, positions)
}
