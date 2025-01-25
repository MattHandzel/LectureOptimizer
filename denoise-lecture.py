import argparse
import logging
import os
import sys
import numpy as np
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
from pydub.silence import detect_silence
from pyrnnoise import pyrnnoise
import moviepy as mpy
import soundfile as sf
import whisper
import librosa


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process video by speeding up silent parts, denoising audio, and generating transcript."
    )
    parser.add_argument("input_video", type=str, help="Path to the input video file")
    parser.add_argument("output_video", type=str, help="Path to the output video file")
    parser.add_argument(
        "--silence_threshold",
        type=float,
        default=-45.0,
        help="Silence threshold in dBFS (default: -40)",
    )
    parser.add_argument(
        "--normal_speed",
        type=float,
        default=1.5,
        help="Speed factor for silent parts (default: 3)",
    )
    parser.add_argument(
        "--silent_speed",
        type=float,
        default=4.0,
        help="Speed factor for silent parts (default: 3)",
    )
    parser.add_argument(
        "--min_silence_len",
        type=int,
        default=250,
        help="Minimum silence length in milliseconds (default: 500)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=200,
        help="Padding around silent regions in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--transcript_output",
        type=str,
        default="transcript.txt",
        help="Path to save the transcript (default: transcript.txt)",
    )
    return parser.parse_args()


def resample_audio(audio_segment, target_frame_rate=48000):
    """Resample audio to target frame rate"""
    return audio_segment.set_frame_rate(target_frame_rate)


def denoise_audio(audio_path, output_path):
    """Denoise audio using RNNoise (requires RNNoise library)"""
    try:
        from pyrnnoise import RNNoise
    except ImportError:
        logger.error("RNNoise library not found. Please install it.")
        sys.exit(1)

    RNNoise = RNNoise()
    audio = AudioSegment.from_file(audio_path)

    return denoised_audio


def process_video(args):
    """Main processing function"""
    # Check input file
    if not os.path.isfile(args.input_video):
        raise FileNotFoundError(f"Input file {args.input_video} not found")

    # Load video
    video = mpy.VideoFileClip(args.input_video)
    original_duration = video.duration
    original_fps = video.fps  # Get original FPS
    logger.info(f"Loaded video: {original_duration:.2f}s duration, {original_fps} FPS")

    # Extract and save original audio
    with NamedTemporaryFile(suffix=".wav", delete=False) as orig_audio_file:
        video.audio.write_audiofile(orig_audio_file.name, logger=None)
        temp_audio_path = orig_audio_file.name

    denoised_audio_path = temp_audio_path  # Default to original if denoising is off
    DENOISE_AUDIO = False
    if DENOISE_AUDIO:
        # Denoise audio
        with NamedTemporaryFile(suffix=".wav", delete=False) as denoised_file:
            logger.info("Denoising audio...")
            info = sf.info(temp_audio_path)
            denoiser = pyrnnoise.RNNoise(info.samplerate, info.channels)

            for _ in denoiser.process_wav(
                temp_audio_path, denoised_file.name
            ):  # it uses a generator
                pass
            denoised_audio_path = denoised_file.name

    # Load denoised/original audio for processing
    audio = AudioSegment.from_file(denoised_audio_path)
    logger.info(f"Loaded audio: {len(audio)/1000:.2f}s")

    # Detect silence
    logger.info("Detecting silent regions...")
    silence_ranges = detect_silence(
        audio,
        min_silence_len=args.min_silence_len,
        silence_thresh=args.silence_threshold,
        seek_step=10,
    )

    # Add padding to silence regions
    silence_ranges = [
        (max(0, start - args.padding), min(len(audio), end + args.padding))
        for (start, end) in silence_ranges
    ]

    # Create segments
    segments = []
    prev_end = 0
    for start, end in silence_ranges:
        if start > prev_end:
            segments.append(("normal", prev_end, start))
        segments.append(("silent", start, end))
        prev_end = end
    if prev_end < len(audio):
        segments.append(("normal", prev_end, len(audio)))

    # Process video and audio segments
    logger.info("Processing video and audio segments...")
    video_clips = []
    audio_segments = []

    normal_effect = mpy.video.fx.MultiplySpeed(factor=args.normal_speed)
    silence_effect = mpy.video.fx.MultiplySpeed(factor=args.silent_speed)

    for seg_type, start, end in segments:
        start_sec = start / 1000.0
        end_sec = end / 1000.0
        assert start_sec < end_sec

        # Process video segment
        clip = video.subclipped(start_sec, end_sec).with_fps(original_fps)
        if seg_type == "silent":
            clip = silence_effect.copy().apply(clip)
            new_duration = (end_sec - start_sec) / args.silent_speed
        else:
            clip = normal_effect.copy().apply(clip)
            new_duration = (end_sec - start_sec) / args.normal_speed
        clip = clip.with_duration(new_duration).with_fps(original_fps)

        # Process audio segment with pitch preservation
        audio_segment = audio[int(start_sec * 1000) : int(end_sec * 1000)]
        # Convert audio to numpy array for processing
        samples = audio_segment.get_array_of_samples()
        channels = audio_segment.channels
        sample_rate = audio_segment.frame_rate
        sample_width = audio_segment.sample_width

        # Convert to numpy array and normalize
        samples_np = np.array(samples, dtype=np.int16)
        samples_float = samples_np.astype(np.float32) / (2**15)

        # Reshape for librosa (channels, samples)
        if channels > 1:
            samples_float = samples_float.reshape((-1, channels)).T
        else:
            samples_float = samples_float.reshape(1, -1)

        # Apply time stretching
        if seg_type == "silent":
            stretched_float = librosa.effects.time_stretch(
                samples_float, rate=args.silent_speed
            )
        else:
            stretched_float = librosa.effects.time_stretch(
                samples_float, rate=args.normal_speed
            )

        # Reshape back to interleaved format
        if channels > 1:
            stretched_float = stretched_float.T.reshape(-1)
        else:
            stretched_float = stretched_float.flatten()

        # Convert back to AudioSegment
        stretched_samples = (stretched_float * (2**15)).astype(np.int16)
        audio_segment = AudioSegment(
            stretched_samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=sample_width,
            channels=channels,
        )

        audio_segments.append(audio_segment)
        video_clips.append(clip)

    # Concatenate video clips
    final_video = mpy.concatenate_videoclips(video_clips).with_fps(original_fps)

    # Concatenate and save processed audio
    processed_audio = (
        sum(audio_segments[1:], audio_segments[0])
        if audio_segments
        else AudioSegment.empty()
    )
    with NamedTemporaryFile(suffix=".wav", delete=False) as processed_audio_file:
        processed_audio.export(processed_audio_file.name, format="wav")
        processed_audio_path = processed_audio_file.name

    # Set processed audio to final video
    final_audio = mpy.AudioFileClip(processed_audio_path)
    final_video = final_video.with_audio(final_audio)
    logger.info(f"Video duration {final_video.duration}")
    logger.info(f"Audio duration {final_audio.duration}")

    logger.info(f"Writing output video to {args.output_video}...")
    final_video.write_videofile(
        args.output_video,
        codec="libx264",
        audio_codec="aac",
        fps=original_fps,
        preset="medium",
        threads=4,
    )

    # Cleanup temporary files
    os.unlink(temp_audio_path)
    if DENOISE_AUDIO:
        os.unlink(denoised_audio_path)
    os.unlink(processed_audio_path)
    video.close()
    final_video.close()


if __name__ == "__main__":
    args = parse_args()
    process_video(args)
    logger.info("Processing completed successfully")
