"""
This file uses the MeloTS model to clone a voice from a reference audio file and a transcript.
"""

import argparse
import os
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from pydub import AudioSegment
import multiprocessing

encoded_message = "MattHandzel"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clone voice from transcript and reference audio."
    )
    parser.add_argument(
        "--tsv_path", type=str, required=True, help="Path to the input TSV file."
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to the reference audio (MP3).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output audio file.",
    )
    parser.add_argument(
        "--ckpt_base",
        type=str,
        default="checkpoints/base_speakers/EN",
        help="Path to base speaker checkpoints.",
    )
    parser.add_argument(
        "--ckpt_converter",
        type=str,
        default="checkpoints/converter",
        help="Path to converter checkpoints.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers.")
    return parser.parse_args()


def read_tsv(tsv_path):
    segments = []
    with open(tsv_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if parts[0] == "start":
                continue

            if len(parts) == 2:
                continue
            assert len(parts) == 3, f"Line {i+1} has {len(parts)} parts."

            if len(parts) == 3:
                start = int(parts[0])
                end = int(parts[1])
                text = parts[2]
                segments.append((start, end, text))
    return segments


# Define global variables for worker processes
worker_base_speaker_tts = None
worker_tone_color_converter = None
worker_target_se = None
worker_use_vad = None
worker_encoded_message = None
worker_target_speed = None
worker_output_dir = None


def init_worker(
    ckpt_base,
    ckpt_converter,
    device_str,
    target_se_cpu,
    use_vad,
    encoded_message,
    target_speed,
    output_dir,
):
    global worker_base_speaker_tts, worker_tone_color_converter, worker_target_se, worker_use_vad, worker_encoded_message, worker_target_speed, worker_output_dir

    device = device_str

    # Initialize models
    worker_base_speaker_tts = BaseSpeakerTTS(f"{ckpt_base}/config.json", device=device)
    worker_base_speaker_tts.load_ckpt(f"{ckpt_base}/checkpoint.pth")

    worker_tone_color_converter = ToneColorConverter(
        f"{ckpt_converter}/config.json", device=device
    )
    worker_tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

    # Convert target_se to appropriate device
    if isinstance(target_se_cpu, torch.Tensor):
        worker_target_se = target_se_cpu.to(device)
    else:
        worker_target_se = torch.tensor(target_se_cpu, device=device)

    worker_use_vad = use_vad
    worker_encoded_message = encoded_message
    worker_target_speed = target_speed
    worker_output_dir = output_dir


def process_segment(segment):
    start, end, text = segment

    src_path = os.path.join(worker_output_dir, f"tmp_{start}.wav")
    if not os.path.exists(src_path):

        worker_base_speaker_tts.tts(
            text,
            src_path,
            speaker="default",
            language="english",
            speed=worker_target_speed,
        )

    # Extract source tone embedding
    source_se, _ = se_extractor.get_se(
        src_path, worker_tone_color_converter, vad=worker_use_vad
    )

    # Convert tone
    converted_path = os.path.join(worker_output_dir, f"converted_{start}.wav")
    if not os.path.exists(converted_path):
        worker_tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=worker_target_se,
            output_path=converted_path,
            message=worker_encoded_message,
        )

    # Load and adjust audio duration
    audio = AudioSegment.from_wav(converted_path)
    desired_duration = end - start
    actual_duration = len(audio)

    if actual_duration > desired_duration:
        audio = audio[:desired_duration]
    elif actual_duration < desired_duration:
        silence = AudioSegment.silent(desired_duration - len(audio))
        audio += silence

    return (start, end, audio)


def clone_voice(
    tsv_path,
    audio_path,
    output_path,
    ckpt_base="checkpoints/base_speakers/EN",
    ckpt_converter="checkpoints/converter",
    num_workers=8,
    encoded_message="MattHandzel",
    use_vad=True,
    target_speed=1.0,
):
    segments = read_tsv(tsv_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize models in main process (for target_se extraction)
    base_speaker_tts = BaseSpeakerTTS(f"{ckpt_base}/config.json", device=device)
    base_speaker_tts.load_ckpt(f"{ckpt_base}/checkpoint.pth")

    tone_color_converter = ToneColorConverter(
        f"{ckpt_converter}/config.json", device=device
    )
    tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

    # Extract target tone embedding
    target_se, _ = se_extractor.get_se(audio_path, tone_color_converter, vad=use_vad)

    # Prepare data for worker processes
    target_se_cpu = (
        target_se.cpu() if isinstance(target_se, torch.Tensor) else target_se
    )

    # Configure parallel processing
    init_args = (
        ckpt_base,
        ckpt_converter,
        device,
        target_se_cpu,
        use_vad,
        encoded_message,
        target_speed,
        output_dir,
    )
    num_workers = min(num_workers, os.cpu_count())

    with multiprocessing.Pool(
        processes=num_workers, initializer=init_worker, initargs=init_args
    ) as pool:
        audio_segments = pool.map(process_segment, segments)

    # Combine audio segments
    if not audio_segments:
        raise ValueError("No audio segments generated.")

    max_end = max(end for _, end, _ in audio_segments)
    frame_rate = audio_segments[0][2].frame_rate
    final_audio = AudioSegment.silent(max_end, frame_rate=frame_rate)

    for start, _, audio in audio_segments:
        final_audio = final_audio.overlay(audio, position=start)

    final_audio.export(output_path, format="mp3")

    return audio_segments


def main():
    args = parse_args()
    clone_voice(
        tsv_path=args.tsv_path,
        audio_path=args.audio_path,
        output_path=args.output_path,
        ckpt_base=args.ckpt_base,
        ckpt_converter=args.ckpt_converter,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
