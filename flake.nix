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
        # GUI support packages
        pkgs.gtk2
        pkgs.gtk3
        pkgs.pkg-config
        pkgs.glib
        pkgs.cairo
        pkgs.pango
        pkgs.gdk-pixbuf
        pkgs.xorg.libX11
        pkgs.xorg.libXext
        pkgs.xorg.libXrender
        pkgs.xorg.libXtst
        pkgs.xorg.libXi
        pkgs.xorg.libXrandr
        
        # OpenCV and its dependencies
        pkgs.python312Packages.opencv-python
        
        # RealSense dependencies
        pkgs.libusb1
        pkgs.udev
        
        # GUI and Audio dependencies
        pkgs.python312Packages.tkinter
        pkgs.python312Packages.pillow
        pkgs.python312Packages.sounddevice
        pkgs.python312Packages.pydub
        pkgs.python312Packages.transformers
        pkgs.python312Packages.torch
        pkgs.python312Packages.torchaudio
        
        # Existing packages
        pkgs.vlc
        pkgs.mpv
        pkgs.ffmpeg
        pkgs.libva
        pkgs.vaapiVdpau
        pkgs.vdpauinfo
        pkgs.python312Full
        pkgs.python312Packages.setuptools
        pkgs.python312Packages.wheel
        pkgs.python312Packages.ultralytics
        pkgs.python312Packages.torchvision
        pkgs.python312Packages.numpy
        pkgs.python312Packages.pyrealsense2
      ];

      shellHook = ''
        echo "Python environment with OpenCV and Ultralytics is ready."
        # Add udev rules for RealSense camera
        export UDEV_RULES_PATH="${pkgs.udev}/lib/udev/rules.d"
      '';
    };
  };
}
