#! /usr/bin/env bash

# whisper input/lecture-4.mp4  --model turbo --threads 15 --language en --output_dir no_denoise/ --word_timestamps True --highlight_words True
# whisper output/lecture-4/lecture-4_denoised.wav  --model turbo --threads 15 --language en --output_dir yes_denoise --word_timestamps True --highlight_words True
# whisper input/lecture-8.mp4  --model turbo --threads 15 --language en --output_dir transcripts/ --word_timestamps True --highlight_words True
#
# py optimize_lecture.py ./input/lecture-6.mp4 --output_dir ./output/lecture-6/ --speed_up --fps 2 --num_workers 15 --overwrite --normal_speed 2 --silent_speed 10 --overwrite --denoise || echo "Failed to process lecture-6"
# py optimize_lecture.py ./input/lecture-7.mp4 --output_dir ./output/lecture-7/ --speed_up --fps 2 --num_workers 15 --overwrite --normal_speed 2 --silent_speed 10 --overwrite --denoise || echo "Failed to process lecture-7"
# py optimize_lecture.py ./input/lecture-8.mp4 --output_dir ./output/lecture-8/ --speed_up --fps 2 --num_workers 15 --overwrite --normal_speed 2 --silent_speed 10 --overwrite --denoise || echo "Failed to process lecture-8"
#

# py 

python optimize_lecture.py ./input/ECE\ 428CS\ 425\ Spring\ 2025\ -\ Illinois\ Media\ Space-07.mp4   --output_dir ./output/CS425-lecture-7/ --speed_up --fps 30 --num_workers 15 --normal_speed 1 --silent_speed 10 --padding 250 --silence_threshold -48.5 --min_silence_len 1000 --denoise

whisper "./output/CS425-lecture-7/CS425-lecture-7_denoised.wav"  --model turbo --threads 15 --language en --output_dir ./output/CS425-lecture-7/ --word_timestamps True --highlight_words True
whisper "./input/ECE 428CS 425 Spring 2025 - Illinois Media Space-07.mp4"  --model turbo --threads 15 --language en --output_dir ./transcripts/ --word_timestamps True --highlight_words True
