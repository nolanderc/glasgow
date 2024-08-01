
# glasgow

> A Language Server for WGSL (WebGPU Shading Language)


## Features

- Completions:
    - Local functions/variables/types.
    - Fields and swizzles.
    - Builtin types and functions (`dot`, `reflect`, `textureSample`, `vec3`, `mat4x2`, etc.)
- Hover Documentation:
    - Function signatures.
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


### Visual Studio Code

Install the `glasgow` extension from the
[marketplace](https://marketplace.visualstudio.com/items?itemName=nolanderc.glasgow).


### neovim

First, install [nvim-lspconfig](https://github.com/neovim/nvim-lspconfig).

Then it is as simple as enabling the `glasgow` configuration:

```lua
local lspconfig = require 'lspconfig'
lspconfig.glasgow.setup {}
```

