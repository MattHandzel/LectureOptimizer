import argparse
import logging
import os
import sys
import numpy as np
import json
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
from pydub.silence import detect_silence
from pyrnnoise import pyrnnoise
import moviepy as mpy
import soundfile as sf
import whisper
import librosa
import cv2
import pytesseract
from pytesseract import TesseractNotFoundError
import torch

global clone_voice
clone_voice = None
import matplotlib.pyplot as plt


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

TARGET_LANGUAGE = "en"

MAX_VOLUME = 0.6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process video by speeding up silent parts, denoising audio, generating transcript, and detecting slide changes with OCR."
    )
    parser.add_argument("input_video", type=str, help="Path to the input video file")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output dir",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--silence_threshold",
        type=float,
        default=-45.0,
        help="Silence threshold in dBFS (default: -40)",
    )
    parser.add_argument(
        "--normal_speed",
        type=float,
        default=2,
        help="Speed factor for normal parts (default: 2)",
    )
    parser.add_argument(
        "--silent_speed",
        type=float,
        default=8.0,
        help="Speed factor for silent parts (default: 8)",
    )
    parser.add_argument(
        "--cut_silence",
        action="store_true",
        help="When flag is active the silent parts will be cutted out",
    )
    parser.add_argument(
        "--normalize_audio",
        type=float,
        help="The audio level to be normalized in decibles",
        default=None,
    )

    parser.add_argument(
        "--min_silence_len",
        type=int,
        default=500,
        help="Minimum silence length in milliseconds (default: 500)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Padding around silent regions in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--transcript_output",
        type=str,
        default="transcript.tsv",
        help="Path to save the transcript (default: transcript.txt)",
    )
    parser.add_argument(
        "--ocr_output",
        type=str,
        default="ocr_output.json",
        help="Path to save the OCR results (default: ocr_output.json)",
    )
    parser.add_argument(
        "--change_threshold",
        type=float,
        default=0.1,
        help="Threshold for detecting slide changes (0.0 to 1.0, default: 0.1)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite the file in the output path"
    )
    parser.add_argument(
        "--source_voice", type=str, help="Path to the source voice file"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=-1,
        help="Set the fps of the video (set this lower to increase processing speed)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for parallel processing in voice cloning.",
    )
    # Boolean flags to control processing steps
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Enable noise reduction on the audio",
    )
    parser.add_argument(
        "--speed_up",
        action="store_true",
        help="Enable speeding up silent parts of the video",
    )
    parser.add_argument(
        "--extract_slides",
        action="store_true",
        help="Enable slide extraction and OCR",
    )
    parser.add_argument(
        "--clone_voice", action="store_true", help="Enable voice cloning"
    )

    args = parser.parse_args()

    args.output_dir = (
        args.output_dir + "/" if not args.output_dir.endswith("/") else args.output_dir
    )

    video_name_without_extension, extension = os.path.splitext(args.input_video)
    output_video_name_without_extension = (
        args.output_dir + video_name_without_extension.split("/")[-1]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    args.num_workers = min(args.num_workers, os.cpu_count())

    args.output_video = output_video_name_without_extension + extension
    args.output_video_name_without_extension = output_video_name_without_extension
    if args.extract_slides:
        args.ocr_output = output_video_name_without_extension + "_ocr.json"
    if args.clone_voice:
        if args.source_voice is None:
            raise ValueError("Please provide a source voice for voice cloning")
        if not args.transcript_output:
            # if transcript output was not specified then set it to the default
            args.transcript_output = output_video_name_without_extension + ".tsv"

    if args.clone_voice:
        global clone_voice
        from clone_voice import clone_voice

    return args


def detect_slide_changes(video_path, threshold=0.1):
    """Analyzes video every second to detect slide changes and perform OCR using Tesseract."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    prev_frame = None
    ocr_data = []
    current_time = 0.0

    while True:
        # Set to current_time seconds
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))  # downsize to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_frame is not None:
            # Calculate absolute difference
            diff = cv2.absdiff(prev_frame, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            change_percent = np.sum(thresh) / thresh.size

            if change_percent > threshold:
                # Perform OCR on the original frame using Tesseract
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    text = pytesseract.image_to_string(rgb_frame)
                except TesseractNotFoundError as e:
                    logger.error(f"Tesseract error: {e}")
                    cap.release()
                    sys.exit(1)
                ocr_data.append({"timestamp": current_time, "text": text.strip()})
                logger.info(f"Change detected at {current_time:.2f}s: OCR performed")

        prev_frame = gray.copy()
        current_time += 1.0  # next second

    cap.release()
    return ocr_data


def normalize_audio_segment(audio_segment, target_dBFS=-20.0):
    """
    Normalize the audio segment to a target dBFS or until it starts to clip, whichever is lower.

    :param audio_segment: AudioSegment object to be normalized
    :param target_dBFS: Target dBFS level to normalize to
    :return: Normalized AudioSegment object
    """
    change_in_dBFS = target_dBFS - audio_segment.dBFS
    normalized_audio = audio_segment.apply_gain(change_in_dBFS)

    # Ensure no clipping
    if normalized_audio.max_dBFS > 0:
        normalized_audio = audio_segment.apply_gain(-audio_segment.max_dBFS)

    return normalized_audio


def process_video(args):
    """Main processing function"""
    # Check input file
    if not os.path.isfile(args.input_video):
        raise FileNotFoundError(f"Input file {args.input_video} not found")

    # Load video
    video = mpy.VideoFileClip(args.input_video)
    if args.fps > 0:
        video = video.with_fps(args.fps)

    original_duration = video.duration
    original_fps = video.fps  # Get original FPS
    logger.info(f"Loaded video: {original_duration:.2f}s duration, {original_fps} FPS")

    # TODO: Normalize the audio levels
    # TODO: Make it so the audio levels are roughly the same from timestamp to timestamp
    # Extract and save original audio
    with NamedTemporaryFile(suffix=".wav", delete=False) as orig_audio_file:
        video.audio.write_audiofile(orig_audio_file.name, logger=None)
        temp_audio_path = orig_audio_file.name
    if args.normalize_audio:
        # Normalize the audio volume levels

        # Load audio
        audio = AudioSegment.from_file(temp_audio_path)
        audio = normalize_audio_segment(audio, target_dBFS=args.normalize_audio)
        # Save normalized audio
        with NamedTemporaryFile(suffix=".wav", delete=False) as normalized_audio_file:
            audio.export(normalized_audio_file.name, format="wav")
            temp_audio_path = normalized_audio_file.name
        input_audio_path = temp_audio_path

    input_audio_path = temp_audio_path  # Default to original if denoising is off
    if args.denoise:
        # Denoise audio
        denoised_audio_path = args.output_video_name_without_extension + "_denoised.wav"
        print(denoised_audio_path)
        if not os.path.exists(denoised_audio_path):
            try:
                with open(denoised_audio_path, "wb") as denoised_file:
                    logger.info("Denoising audio...")
                    info = sf.info(temp_audio_path)
                    denoiser = pyrnnoise.RNNoise(info.samplerate, info.channels)

                    for _ in denoiser.process_wav(
                        temp_audio_path, denoised_file.name
                    ):  # it uses a generator bruh
                        pass
                    denoised_audio_path = denoised_file.name

                    input_audio_path = denoised_audio_path
            except Exception as e:
                logger.error(f"Error denoising audio: {e}")
                input_audio_path = temp_audio_path
                if os.path.exists(denoised_audio_path):
                    os.unlink(denoised_audio_path)

    if args.clone_voice:
        # Use whisper to extract transcript
        print(args.transcript_output)
        print(f"The path exists: {os.path.exists(args.transcript_output)}")
        if not os.path.exists(args.transcript_output):
            from faster_whisper import WhisperModel

            model = WhisperModel(
                "turbo",
                device="cuda" if torch.cuda.is_available() else "cpu",
                num_workers=args.num_workers,
            )
            logger.info("Generating transcript using Whisper...")
            segments, info = model.transcribe(
                input_audio_path, language=TARGET_LANGUAGE, beam_size=5
            )
            with open(args.transcript_output, "w") as f:
                f.write("start\tend\ttext\n")
                for segment in segments:
                    start_ms = int(segment.start * 1000)
                    end_ms = int(segment.end * 1000)
                    text = segment.text.strip()
                    f.write(f"{start_ms}\t{end_ms}\t{text}\n")
            logger.info(f"Transcript saved to {args.transcript_output}")

        # Generate cloned voice audio
        audio_clonning_path = os.path.join(args.output_dir, "audio_clonning")
        os.makedirs(audio_clonning_path, exist_ok=True)
        cloned_audio_path = os.path.join(audio_clonning_path, "cloned_audio.mp3")
        logger.info("Generating cloned voice audio...")
        clone_voice(
            tsv_path=args.transcript_output,
            audio_path=args.source_voice,
            output_path=cloned_audio_path,
            ckpt_base="checkpoints/base_speakers/EN",
            ckpt_converter="checkpoints/converter",
            num_workers=args.num_workers,
            encoded_message="MattHandzel",
            use_vad=True,
            target_speed=1.2,
        )
        input_audio_path = cloned_audio_path

    # Load denoised/original/cloned audio for processing
    audio = AudioSegment.from_file(input_audio_path)
    logger.info(f"Loaded audio: {len(audio)/1000:.2f}s")

    if args.speed_up:
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
            (max(0, start + args.padding), min(len(audio), end - args.padding))
            for (start, end) in silence_ranges
        ]

        # Ensure start of segment isn't less than end of segment
        silence_ranges = [(start, end) for start, end in silence_ranges if start < end]

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

        normal_sound_speed_effect = mpy.video.fx.MultiplySpeed(factor=args.normal_speed)
        silence_speed_effect_effect = mpy.video.fx.MultiplySpeed(
            factor=args.silent_speed
        )

        for seg_type, start, end in segments:
            if args.cut_silence and seg_type == "silent":
                continue
            start_sec = start / 1000.0
            end_sec = end / 1000.0
            assert start_sec < end_sec

            # Process video segment
            clip = video.subclipped(start_sec, end_sec).with_fps(original_fps)
            if seg_type == "silent":
                clip = silence_speed_effect_effect.copy().apply(clip)
                new_duration = (end_sec - start_sec) / args.silent_speed
            else:
                clip = normal_sound_speed_effect.copy().apply(clip)
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
                silent_volume = 0
                stretched_float = (
                    librosa.effects.time_stretch(samples_float, rate=args.silent_speed)
                    * silent_volume
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

            # Ensure the clip is not louder than the maximum volume
            stretched_float = np.clip(stretched_float, -MAX_VOLUME, MAX_VOLUME)

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
        while (
            os.path.exists(args.output_video) and not args.overwrite
        ):  # todo: improve this
            _path, _ext = os.path.splitext(args.output_video)

            args.output_video = _path + "_1" + _ext

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
        if args.denoise:
            os.unlink(denoised_audio_path)
        os.unlink(processed_audio_path)
        video.close()
        final_video.close()

    if args.extract_slides:
        # Perform slide recognition on the input or output video
        target_video = args.output_video if args.speed_up else args.input_video
        logger.info(f"Detecting slide changes and performing OCR on {target_video}...")
        ocr_data = detect_slide_changes(target_video, args.change_threshold)
        logger.info(f"Found {len(ocr_data)} slide changes")

        # Save OCR data to JSON
        if ocr_data:
            with open(args.ocr_output, "w") as f:
                json.dump(ocr_data, f, indent=2)
            logger.info(f"OCR results saved to {args.ocr_output}")
        else:
            logger.warning("No slide changes detected, OCR results not saved.")


if __name__ == "__main__":
    args = parse_args()
    process_video(args)
    logger.info("Processing completed successfully")
