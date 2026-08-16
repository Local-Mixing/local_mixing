# Gauntlet Python env: numpy (builders) + matplotlib (heatmaps).
# Usage: nix-shell shell-gauntlet.nix --run "python3 gauntlet_heatmap.py ..."
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [ (pkgs.python313.withPackages (ps: [ ps.numpy ps.matplotlib ])) ];
}
