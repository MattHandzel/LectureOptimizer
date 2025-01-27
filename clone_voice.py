import argparse
import os
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from pydub import AudioSegment

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
            assert len(parts) == 3, f"Line {i+1} has {len(parts)} parts."
            if len(parts) == 3:
                start = int(parts[0])
                end = int(parts[1])
                text = parts[2]
                segments.append((start, end, text))
    return segments


import multiprocessing

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

    # todo: , enable_watermark=False
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
    worker_base_speaker_tts.tts(
        text, None, speaker="default", language="english", speed=worker_target_speed
    )

    # Extract source tone embedding
    source_se, _ = se_extractor.get_se(
        src_path, worker_tone_color_converter, vad=worker_use_vad
    )

    # Convert tone
    converted_path = os.path.join(worker_output_dir, f"converted_{start}.wav")
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


def main():
    args = parse_args()
    segments = read_tsv(args.tsv_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize models in main process (for target_se extraction)
    base_speaker_tts = BaseSpeakerTTS(f"{args.ckpt_base}/config.json", device=device)
    base_speaker_tts.load_ckpt(f"{args.ckpt_base}/checkpoint.pth")

    # TODO: enable_watermark=False
    tone_color_converter = ToneColorConverter(
        f"{args.ckpt_converter}/config.json",
        device=device,
    )
    tone_color_converter.load_ckpt(f"{args.ckpt_converter}/checkpoint.pth")

    # Extract target tone embedding
    USE_VAD = True
    target_se, _ = se_extractor.get_se(
        args.audio_path, tone_color_converter, vad=USE_VAD
    )
    TARGET_SPEED = 1.0
    encoded_message = "MattHandzel"  # Define appropriately

    # Prepare data for worker processes
    target_se_cpu = (
        target_se.cpu() if isinstance(target_se, torch.Tensor) else target_se
    )

    # Configure parallel processing
    init_args = (
        args.ckpt_base,
        args.ckpt_converter,
        device,
        target_se_cpu,
        USE_VAD,
        encoded_message,
        TARGET_SPEED,
        output_dir,
    )
    num_workers = min(args.num_workers, os.cpu_count())  # Adjust based on resources

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

    final_audio.export(args.output_path, format="mp3")


if __name__ == "__main__":
    main()
