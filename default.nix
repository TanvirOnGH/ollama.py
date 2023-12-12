# Reference: <https://nixos.wiki/wiki/Python>
{pkgs ? import <nixpkgs> {}}: let
  deps = ps:
    with ps; [
      # For formatting
      black

      requests
    ];
  python = pkgs.python3.withPackages deps; # Python3.11 as of writing
in
  python.env
