{
  description = "Flake: YOLO prototype in a pinned Python env";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }: let
    system = "x86_64-linux";
    pkgs   = import nixpkgs { inherit system; config.allowUnfree = true; };

    pythonEnv = pkgs.python312.withPackages (ps: with ps; [
      spacy
      spacy-models.en_core_web_sm   # Note the dot here
      transformers
      torch
      torchaudio
      ultralytics
      numpy
      pyrealsense2
      opencv-python
      pillow
      sounddevice
      pydub
      # Any other Python packages you need
    ]);
  in {
    devShell.${system} = pkgs.mkShell {
      buildInputs = [
        pythonEnv
        pkgs.gtk2 pkgs.gtk3 pkgs.pkg-config pkgs.glib pkgs.cairo pkgs.pango pkgs.gdk-pixbuf
        pkgs.xorg.libX11 pkgs.xorg.libXext pkgs.xorg.libXrender pkgs.xorg.libXtst pkgs.xorg.libXi pkgs.xorg.libXrandr
        pkgs.libusb1 pkgs.udev pkgs.vlc pkgs.mpv pkgs.ffmpeg pkgs.libva pkgs.vaapiVdpau pkgs.vdpauinfo
      ];

      shellHook = ''
        echo "🐍 Python env with spaCy+model is ready!"
        # Add udev rules for RealSense camera
        export UDEV_RULES_PATH="${pkgs.udev}/lib/udev/rules.d"
      '';
    };
  };
}
