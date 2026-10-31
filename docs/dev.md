## Introduction

Developers for MatCalc should consult this guide for some best practices.

## UV

We use [uv] as our packaging tool to speed up installs and streamline management of dependencies. UV can be installed
using brew or pip.

```shell
brew install uv
```

or

```shell
pip install uv
```

`uv.lock` does not cover Intel Macs, so `uv sync` fails there: deepmd-kit 3.2 requires torch 2.11, which has no Intel
macOS wheels.

Please refer to the [uv] documentation on how to manage dependencies. The core dependencies should be kept to an
absolute minimum. Dependencies that are only used for development should be placed in the `dev`
dependency group using:

```shell
uv add <dependency> --dev
```

Dependencies required for different models should always be optional. For example, matgl is an optional dependency
(as well as a dev dependency used for unit testing) that is used only when you plan to use MatGL models with MatCalc.

```shell
uv add matgl --optional matgl
```

[uv]: https://docs.astral.sh/uv
