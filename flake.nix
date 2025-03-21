{
  description = "Flake: YOLO prototype in a pinned Python env without poetry2nix";

  inputs = {
    # Nixpkgs for base system packages
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
  let
    system = "x86_64-linux";  # or aarch64-linux on ARM
    pkgs = import nixpkgs { 
      inherit system; 
      config.allowUnfree = true;
    };
  in
  {
    # devShell for your development environment
    devShell.${system} = pkgs.mkShell {
      # The main Python environment
      buildInputs = [
        pkgs.vlc
        pkgs.mpv
        pkgs.ffmpeg
        pkgs.libva
        pkgs.vaapiVdpau
        pkgs.vdpauinfo
        pkgs.python312Full
        pkgs.python312Packages.setuptools
        pkgs.python312Packages.wheel
        pkgs.python312Packages.opencv-python
        pkgs.python312Packages.ultralytics
        pkgs.python312Packages.torchvision
        pkgs.python312Packages.torchaudio
        pkgs.python312Packages.numpy
        pkgs.python312Packages.pyrealsense2
        pkgs.python312Packages.torch
      ];

      shellHook = ''
        echo "Python environment with OpenCV and Ultralytics is ready."
      '';
    };
  };
}
