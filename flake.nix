{
  description = "Python environment with dependencies managed via mach-nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    mach-nix.url = "github:DavHau/mach-nix/3.2.0";
  };

  outputs = { self, nixpkgs, flake-utils, mach-nix }:

    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;
        mach-nix = import (builtins.fetchGit {
          url = "https://github.com/DavHau/mach-nix.git";
          ref = "master";  # Or a specific branch/tag
          rev = "";  # Replace with the latest commit hash
        }) { inherit pkgs; };
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;

        # System dependencies required by Python packages
        systemDeps = with pkgs; [
          mecab
          libgcc.lib
          stdenv.cc.cc.lib
        ];

        # Python requirements including version pins
        requirements = ''
          pyannote.audio
          whisper
          pydub
          pyrnnoise
          faster-whisper
          whisper-timestamped
          inflect
          txtsplit
          torch
          torchaudio
          cached-path
          transformers
          num2words==0.5.12
          unidic-lite==1.0.8
          unidic==1.1.0
          mecab-python3==1.0.9
          pykakasi==2.2.1
          fugashi==1.3.0
          g2p-en==2.1.0
          anyascii==0.3.2
          jamo==0.4.1
          gruut[de,es,fr]==2.2.3
          g2pkk>=0.1.1
          librosa==0.9.1
          pydub==0.25.1
          eng-to-ipa==0.0.2
          inflect==7.0.0
          unidecode==1.3.7
          pypinyin==0.50.0
          cn2an==0.5.22
          jieba==0.42.1
          gradio
          langid==1.1.6
          tqdm
          tensorboard==2.16.2
          loguru==0.7.2
        '';

        # Create Python environment with mach-nix
        pythonEnv = mach-nix.lib.${system}.mkPython {
          inherit requirements;
          providers = {
            # Use nixpkgs versions for better compatibility
            torch = "nixpkgs";
            torchaudio = "nixpkgs";
            opencv-python = "nixpkgs";
            librosa = "nixpkgs";
          };
          _."pydub".extras = [ "pyaudioops" ];  # Example of adding extras
        };

      in {
        # Buildable package
        packages.default = pkgs.buildEnv {
          name = "python-environment";
          paths = [ pythonEnv ] ++ systemDeps;
          extraOutputsToInstall = [ "lib" ];
        };

        # Development shell
        devShells.default = pkgs.mkShell {
          buildInputs = [ pythonEnv ] ++ systemDeps;
          shellHook = ''
            export LD_LIBRARY_PATH=${lib.makeLibraryPath systemDeps}:$LD_LIBRARY_PATH
            echo "Python environment ready. Run 'python' to start."
          '';
        };
      });
}
