use std::collections::BTreeMap;

use bstr::BString;

#[derive(Default)]
pub struct Workspace {
    pub documents: BTreeMap<lsp::Uri, Document>,
}

pub struct Document {
    pub content: BString,
}
