{
  description = "tickerkit — unified, provenance-tracked crypto ticker snapshots";

  # Pinned to the revision already in the local store so `nix develop` is instant.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/20b1ddd1aa5ace70c9468305030aa4f9ef79671b";

  # uv2nix stack: builds the uv.lock dependency graph from wheels into a hermetic
  # Nix package. Needed because ccxt (tickerkit's core dependency) is not packaged in
  # nixpkgs, so the runtime deps must come from PyPI — which the lockfile pins exactly.
  inputs.pyproject-nix.url = "github:pyproject-nix/pyproject.nix";
  inputs.pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";
  inputs.uv2nix.url = "github:pyproject-nix/uv2nix";
  inputs.uv2nix.inputs.pyproject-nix.follows = "pyproject-nix";
  inputs.uv2nix.inputs.nixpkgs.follows = "nixpkgs";
  inputs.pyproject-build-systems.url = "github:pyproject-nix/build-system-pkgs";
  inputs.pyproject-build-systems.inputs.pyproject-nix.follows = "pyproject-nix";
  inputs.pyproject-build-systems.inputs.uv2nix.follows = "uv2nix";
  inputs.pyproject-build-systems.inputs.nixpkgs.follows = "nixpkgs";

  outputs =
    { self, nixpkgs, pyproject-nix, uv2nix, pyproject-build-systems }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAll = nixpkgs.lib.genAttrs systems;

      # Python 3.12 to match requires-python and the uv-managed interpreter.
      pythonFor =
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          inherit pkgs;
          python = pkgs.python312;
        };

      # The hermetic package set, built from uv.lock. Shared by `packages` and the
      # helper apps. Use `lib.sources.cleanSource` (gitignore-aware) so the store path
      # never includes .venv / .uv-cache / tickerkit_cache.sqlite3.
      uv2nixFor =
        system:
        let
          inherit (pythonFor system) pkgs python;
          inherit (nixpkgs) lib;
          pythonBase = pkgs.callPackage pyproject-nix.build.packages { inherit python; };
          workspace = uv2nix.lib.workspace.loadWorkspace {
            workspaceRoot = lib.sources.cleanSource ./.;
          };
          overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
          pythonSet = pythonBase.overrideScope (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
            ]
          );
          # A plain package derivation has no working interpreter path for its entry
          # point scripts, so expose the CLI through the documented
          # virtualenv + mkApplication pattern: the venv aggregates every lockfile
          # dependency, and mkApplication exports a small derivation containing only
          # tickerkit's own bin contents with that venv as context.
          inherit (pkgs.callPackages pyproject-nix.build.util { }) mkApplication;
          package = mkApplication {
            venv = pythonSet.mkVirtualEnv "tickerkit-env" workspace.deps.default;
            package = pythonSet.tickerkit;
          };
        in
        {
          inherit pythonSet package;
        };
    in
    {
      devShells = forAll (
        system:
        let
          inherit (pythonFor system) pkgs python;
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
              # Lint/type tools come from nixpkgs rather than the venv so they are
              # available before `uv sync` has ever run. ccxt is not packaged in
              # nixpkgs, which is why the Python *dependencies* still come from uv.
              pkgs.ruff
              pkgs.basedpyright
              pkgs.git
            ];

            # This is the reason the flake exists. `uv` installs PyPI wheels for numpy
            # and pandas, which link against libz.so.1 and libstdc++.so.6 by soname.
            # NixOS has no /lib, so without this the notebook dies at `import pandas`
            # with "libz.so.1: cannot open shared object file".
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.zlib
              pkgs.stdenv.cc.cc.lib
            ];

            shellHook = ''
              # Keep uv's cache inside the project so nothing leaks into $HOME.
              export UV_CACHE_DIR="$PWD/.uv-cache"
              export PYTHONDONTWRITEBYTECODE=1

              if [ ! -x .venv/bin/python ]; then
                echo "tickerkit: creating .venv (first run only)…"
                uv sync --quiet
              fi

              # The kernelspec Zed looks for; pins .venv/bin/python by absolute path.
              if [ -x .venv/bin/python ] && [ ! -e "$HOME/.local/share/jupyter/kernels/tickerkit" ]; then
                uv run python tools/install_kernel.py >/dev/null 2>&1 || true
              fi

              # Put the venv first so `python`, `pytest`, `jupyter` and `ruff` are the
              # project's own, not the interpreter-only Nix python above.
              if [ -d .venv/bin ]; then
                export PATH="$PWD/.venv/bin:$PATH"
              fi

              echo "tickerkit dev shell — python $(python -V 2>&1 | cut -d' ' -f2), uv $(uv --version | cut -d' ' -f2)"
              echo "  uv run pytest -m 'not live'   # offline test suite"
              echo "  uv run python tools/validate_notebook.py"
              echo "  uv run python tools/validate_notebook.py notebook/history.ipynb"
              echo "  uv run jupyter lab"
            '';
          };
        }
      );

      # A sandboxed check must not touch the network, and tickerkit's dependencies
      # (notably ccxt) are not in nixpkgs, so the real test suite cannot run as a Nix
      # build. What *can* be checked offline is the source tree itself: lint and
      # formatting. Use `nix develop -c uv run pytest` for the test suite.
      checks = forAll (
        system:
        let
          inherit (pythonFor system) pkgs;
        in
        {
          # `ruff check` only: the repo is not `ruff format`-clean and the project's
          # documented gate has never been the formatter.
          ruff = pkgs.runCommand "tickerkit-ruff" { nativeBuildInputs = [ pkgs.ruff ]; } ''
            cd ${self}
            # --no-cache: the source tree is a read-only store path, so ruff cannot
            # create its .ruff_cache directory there.
            ruff check --no-cache src/ tests/ examples/ tools/
            touch $out
          '';
        }
      );

      formatter = forAll (system: nixpkgs.legacyPackages.${system}.nixfmt-rfc-style);

      # Hermetic build from uv.lock: `nix build .#default`, `nix run .#default`,
      # or remotely `nix run github:Cairnstew/tickerkit -- BTC`.
      packages = forAll (
        system:
        let
          inherit (uv2nixFor system) package;
        in
        {
          default = package;
          tickerkit = package;
        }
      );

      apps = forAll (
        system:
        let
          inherit (pythonFor system) pkgs python;
        in
        {
          # The hermetic CLI from uv.lock; `nix run .#default -- BTC`.
          default = {
            type = "app";
            program = "${self.packages.${system}.default}/bin/tickerkit";
          };
          tickerkit = {
            type = "app";
            program = "${self.packages.${system}.default}/bin/tickerkit";
          };

          # `nix run .#test` executes on the host, so uv may reach the network here
          # even though a Nix *build* may not.
          test = {
            type = "app";
            program = "${pkgs.writeShellApplication {
              name = "tickerkit-test";
              runtimeInputs = [
                python
                pkgs.uv
              ];
              text = ''
                if [ ! -f pyproject.toml ]; then
                  echo "run this from the tickerkit checkout" >&2
                  exit 1
                fi
                uv sync --quiet
                exec uv run pytest -m "not live" "$@"
              '';
            }}/bin/tickerkit-test";
          };

          notebook = {
            type = "app";
            program = "${pkgs.writeShellApplication {
              name = "tickerkit-notebook";
              runtimeInputs = [
                python
                pkgs.uv
              ];
              text = ''
                uv sync --quiet
                exec uv run python tools/validate_notebook.py
              '';
            }}/bin/tickerkit-notebook";
          };

          # `nix run .#history-notebook` validates notebook/history.ipynb — the deep
          # history lookup + testing notebook.
          history-notebook = {
            type = "app";
            program = "${pkgs.writeShellApplication {
              name = "tickerkit-history-notebook";
              runtimeInputs = [
                python
                pkgs.uv
              ];
              text = ''
                uv sync --quiet
                exec uv run python tools/validate_notebook.py notebook/history.ipynb
              '';
            }}/bin/tickerkit-history-notebook";
          };
        }
      );
    };
}