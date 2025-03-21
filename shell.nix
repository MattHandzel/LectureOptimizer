# pip install pyannote.audio whisper pydub pyrnnoise
{pkgs ? import <nixpkgs> {}}:
with pkgs;
  mkShell {
    buildInputs = with python312Packages;
      [virtualenv qdrant-client python-dotenv flask openai-whisper ffmpeg-python yt-dlp pydub pyannote-audio opencv-python sentence-transformers scikit-learn numpy matplotlib tqdm soundfile librosa pytesseract torch torchvision tokenizers yt-dlp ]
      ++ [
        python312
        libgcc.lib
        mecab # needed for MeloTTS
        ollama
        python312Packages.ollama


        # docker
        # docker-compose
      ];

    shellHook = ''

      export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [pkgs.libgcc.lib]}:$LD_LIBRARY_PATH
             if ! [ -e .venv ]; then
               python3 -m venv .venv
               pip install python-frontmatter anki pyannote.audio whisper pydub pyrnnoise faster_whisper whisper_timestamped inflect txtsplit torch torchaudio cached_path transformers num2words==0.5.12 unidic_lite==1.0.8 unidic==1.1.0 mecab-python3==1.0.9 pykakasi==2.2.1 fugashi==1.3.0 g2p_en==2.1.0 anyascii==0.3.2 jamo==0.4.1 gruut[de,es,fr]==2.2.3 g2pkk>=0.1.1 librosa==0.9.1 pydub==0.25.1 eng_to_ipa==0.0.2 inflect==7.0.0 unidecode==1.3.7 pypinyin==0.50.0 cn2an==0.5.22 jieba==0.42.1 gradio langid==1.1.6 tqdm tensorboard==2.16.2 loguru==0.7.2
             fi
            source .venv/bin/activate


      # export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
      # ollama serve > /dev/null 2>&1 &
            #
    '';
  }
