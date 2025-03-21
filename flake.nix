# WARNING: THIS DOES NOT WORK!!!!

{
  description = "Lecture Optimizer: A tool to optimize educational videos";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;

        # System dependencies required by Python packages
        systemDeps = with pkgs; [
          mecab
          libgcc.lib
          stdenv.cc.cc.lib
          ffmpeg  # Required for audio/video processing
        ];

        # Python packages from nixpkgs
        pythonPackages = with pkgs.python312Packages; [
          virtualenv
          python-dotenv
          openai-whisper
          ffmpeg-python
          yt-dlp
          pydub
          pyannote-audio
          opencv-python
          sentence-transformers
          scikit-learn
          numpy
          matplotlib
          tqdm
          soundfile
          librosa
          pytesseract
          torch
          torchvision
          tokenizers
        ];

        # Include your project's source code
        lectureOptimizer = pkgs.stdenv.mkDerivation {
          name = "lecture-optimizer";
          src = ./.;

          buildInputs = pythonPackages ++ systemDeps;

          installPhase = ''
            mkdir -p $out/bin
            cp optimize_lecture.py $out/bin/optimize_lecture
            chmod +x $out/bin/optimize_lecture
          '';
        };

      in {
        # Buildable package
        packages.default = lectureOptimizer;

        # Development shell
        devShells.default = pkgs.mkShell {
          buildInputs = [ lectureOptimizer ] ++ pythonPackages ++ systemDeps;
          shellHook = ''
            export LD_LIBRARY_PATH=${lib.makeLibraryPath systemDeps}:$LD_LIBRARY_PATH
            echo "Lecture Optimizer environment ready. Run 'optimize_lecture' to start."
          '';
        };
      });
}
