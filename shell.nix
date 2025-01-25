# pip install pyannote.audio whisper pydub pyrnnoise
{pkgs ? import <nixpkgs> {}}:
with pkgs;
  mkShell {
    buildInputs = with python312Packages;
      [virtualenv qdrant-client python-dotenv flask openai-whisper ffmpeg-python yt-dlp pydub pyannote-audio opencv-python sentence-transformers scikit-learn numpy matplotlib tqdm soundfile librosa]
      ++ [
        python312
        # ollama
        # python312Packages.ollama

        docker
        docker-compose
      ];

    shellHook = ''
       if ! [ -e .venv ]; then
         python3 -m venv .venv
         pip install python-frontmatter anki
       fi
      source .venv/bin/activate


      # export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
      # ollama serve > /dev/null 2>&1 &
      #
    '';
  }
