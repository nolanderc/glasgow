
# glasgow

> A Language Server for WGSL (WebGPU Shading Language)


## Features

- Completions:
    - Local functions/variables/types.
    - Fields and swizzles.
    - Builtin types and functions (`dot`, `reflect`, `textureSample`, `vec3`, `mat4x2`, etc.)
- Hover Documentation:
    - Funtion signatures.
    - Variable types.
    - Includes builtin types and functions. Text is taken from the WGSL specification.
- Goto Definition
- Find all References
- Rename
- Formatter


### Planned

- Support for non-standard `#include`/`#import` directives.


## Usage

First install the language server using `cargo`:

```sh
cargo install glasgow
```

Then follow the editor-specific instructions below:

### neovim

First, install [nvim-lspconfig](https://github.com/neovim/nvim-lspconfig).

As `glasgow` has not yet been included in the default configs, it has to be
added it manually:

```lua
local lspconfig = require 'lspconfig'
require('lspconfig.configs').glasgow = {
    default_config = {
        cmd = {'glasgow'},
        filetypes = {'wgsl'},
        root_dir = lspconfig.util.find_git_ancestor,
        single_file_support = true,
        settings = {},
    }
}
lspconfig.glasgow.setup { capabilities = capabilities, on_attach = on_attach }
```

