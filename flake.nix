{
  inputs = {

    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
    devenv.url = "github:cachix/devenv";
    devenv.inputs.nixpkgs.follows = "nixpkgs";
    fenix.url = "github:nix-community/fenix";
    fenix.inputs = { nixpkgs.follows = "nixpkgs"; };
  };

  outputs = { self, nixpkgs, utils, devenv, ... }@inputs:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        packages = {
          devenv-up = self.devShell.${system}.config.procfileScript;
        };
        devShell = devenv.lib.mkShell {
          inherit inputs pkgs;
          modules = [
            ({ pkgs, config, ... }: with pkgs; {

              packages = [
                openssl
                pkg-config
                libclang.lib
                llvmPackages.libcxxClang
                clang
                gnum4
              ];
              hardeningDisable = [ "fortify" ];

              languages.rust =
                {
                  channel = "nightly";
                  enable = true;
                };
              env.LIBCLANG_PATH = "${libclang.lib}/lib";
              env.BINDGEN_EXTRA_CLANG_ARGS = "-isystem ${llvmPackages.libcxxClang}/lib/clang/${lib.getVersion clang}/include";

              pre-commit.hooks = {
                nixpkgs-fmt.enable = true;
                rustfmt.enable = true;
              };
            })
          ];
        };
      });
}

