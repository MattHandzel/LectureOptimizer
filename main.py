# lecture_analyzer.py
import os
import sys
import logging
import argparse
import subprocess
import numpy as np
import cv2
import ffmpeg
from typing import List, Dict, Tuple, Optional
from pydub import AudioSegment
from pyrnnoise import RNNoise
import whisper
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio import Pipeline
from dataclasses import dataclass
from tqdm import tqdm
from dotenv import load_dotenv

# TODO: Audio equalizer

try:
    import matplotlib.pyplot as plt
    CAN_PLOT = True
except ImportError:
    CAN_PLOT = False

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    # Video parameters
    chunk_size: int = 500  # MB
    chunk_duration: str = "00:10:00"
    output_dir: str = "output"

    # Audio parameters
    sample_rate: int = 16000
    denoise_enabled: bool = True
    denoise_chunk_size: int = 480

    # Transcription parameters
    whisper_model: str = "base"
    language: Optional[str] = None
    beam_size: int = 5

    # Analysis parameters
    text_window_size: int = 3
    text_threshold: float = 0.4
    slide_threshold: float = 0.2
    min_slide_interval: float = 2.0

    # Visualization parameters
    visualize: bool = False
    visualize_dir: str = "visualizations"
    show_plots: bool = False

    # System parameters
    debug: bool = False
    keep_files: bool = False
    diarization_enabled: bool = False

class EnhancedVideoProcessor:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)

    def download_video(self, url: str, platform: str) -> Tuple[Optional[str], Optional[str]]:
        """Download video with progress tracking and format selection"""
        logger.info(f"Starting download from {platform}: {url}")

        try:
            cmd = [
                "yt-dlp",
                "-o",
                f"{self.config.output_dir}/%(title)s.%(ext)s",
                "-f",
                "worstvideo[ext=mp4]+bestaudio[ext=m4a]/worst[ext=mp4]/worst",
                "--newline",
                "--progress",
                url,
            ]

            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
            ) as proc:
                filename = None
                progress_bar = None

                for line in proc.stdout:
                    if self.config.debug:
                        logger.debug(f"yt-dlp: {line.strip()}")
                    
                    if "Destination: " in line:
                        filename = line.split("Destination: ")[1].strip()
                    elif "[download]" in line and "%" in line:
                        if not progress_bar:
                            total = 100.0
                            progress_bar = tqdm(
                                total=total,
                                unit="%",
                                desc="Downloading",
                                disable=not self.config.debug,
                            )
                        percent = float(line.split("%")[0].split()[-1])
                        progress_bar.update(percent - progress_bar.n)

                if progress_bar:
                    progress_bar.close()

                return filename, None

        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            return None, str(e)

    def chunk_video(self, input_path: str) -> List[str]:
        """Split video into chunks with progress tracking"""
        logger.info(f"Chunking video: {input_path}")
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_pattern = f"{self.config.output_dir}/{base_name}_%03d.mp4"

        try:
            # Get duration using ffprobe
            duration = float(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        input_path,
                    ]
                ).decode().strip()
            )

            progress_bar = tqdm(
                total=duration, 
                unit="s", 
                desc="Chunking", 
                disable=not self.config.debug
            )

            # Run ffmpeg with progress parsing
            process = (
                ffmpeg.input(input_path)
                .output(
                    output_pattern,
                    c="copy",
                    map="0",
                    segment_time=self.config.chunk_duration,
                    f="segment",
                    reset_timestamps="1",
                )
                .global_args("-progress", "pipe:1")
                .overwrite_output()
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )

            while True:
                line = process.stdout.readline()
                if not line:
                    break
                if "out_time_ms" in line:
                    time_ms = int(line.split("=")[1].strip())
                    progress_bar.update(time_ms / 1e6 - progress_bar.n)

            process.wait()
            progress_bar.close()

            chunks = sorted(
                [
                    f"{self.config.output_dir}/{f}"
                    for f in os.listdir(self.config.output_dir)
                    if f.startswith(base_name) and f.endswith(".mp4")
                ]
            )

            logger.info(f"Created {len(chunks)} video chunks")
            return chunks

        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise RuntimeError("Video chunking failed") from e
        finally:
            if 'progress_bar' in locals():
                progress_bar.close()

class EnhancedAudioProcessor:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.denoiser = None
        if config.denoise_enabled:
            try:
                # RNNoise requires 48kHz input
                self.denoiser = RNNoise(sample_rate=48000)
            except Exception as e:
                logger.error(f"Failed to initialize RNNoise: {str(e)}")
                raise RuntimeError("Audio denoising unavailable") from e

    def extract_audio(self, video_path: str) -> str:
        """Extract audio with progress tracking"""
        logger.debug(f"Extracting audio from {video_path}")
        audio_path = f"{os.path.splitext(video_path)[0]}.wav"

        try:
            (
                ffmpeg.input(video_path)
                .output(audio_path, ar=str(self.config.sample_rate), ac=1, ab="192k")
                .overwrite_output()
                .run(quiet=not self.config.debug)
            )
            return audio_path
        except ffmpeg.Error as e:
            logger.error(f"Audio extraction failed: {e.stderr.decode()}")
            raise

    def denoise_audio(self, audio_path: str) -> str:
        """Denoise audio with RNNoise"""
        if not self.config.denoise_enabled or not self.denoiser:
            logger.info("Skipping denoising as configured")
            return audio_path

        logger.info(f"Denoising audio: {audio_path}")
        audio = AudioSegment.from_wav(audio_path)
        
        # Resample to 48kHz if necessary
        if audio.frame_rate != 48000:
            audio = audio.set_frame_rate(48000)
            
        samples = np.array(audio.get_array_of_samples())

        denoised = []
        progress = tqdm(
            total=len(samples),
            unit="samples",
            desc="Denoising",
            disable=not self.config.debug,
        )

        try:
            for i in range(0, len(samples), self.config.denoise_chunk_size):
                chunk = samples[i:i + self.config.denoise_chunk_size].astype(np.float32) / 32768.0
                denoised_chunk = self.denoiser.process_chunk(chunk)
                denoised.append(denoised_chunk * 32768.0)
                progress.update(len(chunk))

            denoised_samples = np.concatenate(denoised).astype(np.int16)
            denoised_path = audio_path.replace(".wav", "_denoised.wav")

            AudioSegment(
                denoised_samples.tobytes(),
                frame_rate=48000,  # Maintain denoiser sample rate
                sample_width=2,
                channels=1,
            ).export(denoised_path, format="wav")

            return denoised_path
        except Exception as e:
            logger.error(f"Denoising failed: {str(e)}")
            raise
        finally:
            progress.close()

class EnhancedTranscriptGenerator:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        logger.info(f"Loading Whisper model: {config.whisper_model}")
        self.model = whisper.load_model(config.whisper_model)

    def transcribe(self, audio_path: str) -> List[Dict]:
        """Generate transcript with configurable parameters"""
        logger.info(f"Transcribing audio: {audio_path}")

        try:
            result = self.model.transcribe(
                audio_path,
                language=self.config.language,
                beam_size=self.config.beam_size,
                word_timestamps=True,
                verbose=self.config.debug,
            )

            return [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                    "words": [
                        {"word": w["word"], "start": w["start"], "end": w["end"]}
                        for w in s.get("words", [])
                    ],
                }
                for s in result["segments"]
            ]
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

class EnhancedVisualizer:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        os.makedirs(config.visualize_dir, exist_ok=True)

    def visualize_analysis(self, analysis: Dict, chunk_name: str):
        """Generate visualizations if enabled"""
        if not self.config.visualize or not CAN_PLOT:
            return

        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        self._plot_timeline(axs[0], analysis)
        self._plot_word_distribution(axs[1], analysis)
        plt.suptitle(f"Analysis Visualization - {chunk_name}")
        plt.tight_layout()

        if self.config.show_plots:
            plt.show()
        else:
            save_path = os.path.join(
                self.config.visualize_dir, f"{chunk_name}_analysis.png"
            )
            plt.savefig(save_path)
            logger.info(f"Saved visualization to {save_path}")
        plt.close()

    def _plot_timeline(self, ax, analysis):
        ax.set_title("Content Timeline")
        for topic in analysis["text_topics"]:
            ax.plot(
                [topic["start"], topic["end"]],
                [0, 0],
                linewidth=10,
                label=f"Topic {topic['topic_id']}",
            )
        for change in analysis["slide_changes"]:
            ax.axvline(change, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel("Time (seconds)")
        ax.set_yticks([])
        ax.legend()

    def _plot_word_distribution(self, ax, analysis):
        words = [
            w["word"] for segment in analysis["transcript"] for w in segment["words"]
        ]
        unique, counts = np.unique(words, return_counts=True)
        ax.barh(unique[np.argsort(-counts)[:20]], np.sort(counts)[::-1][:20])
        ax.set_title("Top 20 Frequent Words")
        ax.set_xlabel("Count")

class LectureAnalyzer:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.video_processor = EnhancedVideoProcessor(config)
        self.audio_processor = EnhancedAudioProcessor(config)
        self.transcriber = EnhancedTranscriptGenerator(config)
        self.visualizer = EnhancedVisualizer(config)

        # Initialize diarization pipeline if needed
        self.diarization_pipeline = None
        if config.diarization_enabled:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError("Hugging Face token required for diarization")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization", 
                use_auth_token=hf_token
            )

    def process_lecture(self, source: str, platform: str) -> List[Dict]:
        """Full processing pipeline with error handling"""
        chunks = []
        try:
            # 1. Acquire video
            if platform != "local":
                logger.info(f"Downloading from {platform}...")
                video_path, error = self.video_processor.download_video(source, platform)
                if error:
                    raise RuntimeError(f"Download failed: {error}")
            else:
                video_path = source
                logger.info(f"Processing local file: {video_path}")

            # 2. Chunk video
            chunks = self.video_processor.chunk_video(video_path)
            full_analysis = []

            # 3. Process each chunk
            for chunk in tqdm(
                chunks, 
                desc="Processing chunks", 
                disable=not self.config.debug
            ):
                chunk_analysis = self._process_chunk(chunk)
                full_analysis.append(chunk_analysis)
                self.visualizer.visualize_analysis(chunk_analysis, os.path.basename(chunk))

            return full_analysis

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            if self.config.debug:
                logger.exception("Full error traceback:")
            raise
        finally:
            if not self.config.keep_files:
                self._cleanup_temp_files(chunks)

    def _process_chunk(self, chunk_path: str) -> Dict:
        """Process individual video chunk"""
        chunk_analysis = {"chunk": chunk_path}
        audio_path = None
        clean_audio = None

        try:
            # Audio processing
            audio_path = self.audio_processor.extract_audio(chunk_path)
            clean_audio = self.audio_processor.denoise_audio(audio_path)

            # Transcription
            transcript = self.transcriber.transcribe(clean_audio)
            chunk_analysis["transcript"] = transcript

            # Content analysis
            chunk_analysis.update(self._analyze_content(chunk_path, transcript))
            return chunk_analysis

        finally:
            if not self.config.keep_files:
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
                if clean_audio and clean_audio != audio_path and os.path.exists(clean_audio):
                    os.remove(clean_audio)

    def _analyze_content(self, chunk_path: str, transcript: List[Dict]) -> Dict:
        """Perform text and visual analysis"""
        analysis = {}

        # Text analysis
        if transcript:
            text_model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = text_model.encode([t["text"] for t in transcript])
            time_matrix = self._create_time_matrix([t["start"] for t in transcript])
            combined_sim = 1 - np.dot(embeddings, embeddings.T) * time_matrix

            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.config.text_threshold,
                affinity="precomputed",
                linkage="average",
            ).fit(combined_sim)

            analysis["text_topics"] = self._format_clusters(clustering.labels_, transcript)
        else:
            analysis["text_topics"] = []

        # Slide change detection
        analysis["slide_changes"] = self._detect_slide_changes(chunk_path)
        return analysis

    def _create_time_matrix(self, timestamps: List[float]) -> np.ndarray:
        time_diffs = np.abs(np.array(timestamps)[:, None] - np.array(timestamps))
        return 1 / (1 + np.exp(0.1 * time_diffs))  # Fixed time weighting

    def _format_clusters(self, labels: np.ndarray, transcript: List[Dict]) -> List[Dict]:
        clusters = []
        if not transcript:
            return clusters

        current_cluster = [transcript[0]]
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                clusters.append({
                    "start": current_cluster[0]["start"],
                    "end": current_cluster[-1]["end"],
                    "text": " ".join([t["text"] for t in current_cluster]),
                    "topic_id": labels[i-1],
                })
                current_cluster = [transcript[i]]
            else:
                current_cluster.append(transcript[i])

        # Add the last cluster
        if current_cluster:
            clusters.append({
                "start": current_cluster[0]["start"],
                "end": current_cluster[-1]["end"],
                "text": " ".join([t["text"] for t in current_cluster]),
                "topic_id": labels[-1] if len(labels) > 0 else 0,
            })
        return clusters

    def _detect_slide_changes(self, video_path: str) -> List[float]:
        """Detect visual slide changes with progress tracking"""
        logger.debug(f"Detecting slide changes in {video_path}")
        cap = cv2.VideoCapture(video_path)
        changes = []
        prev_frame = None
        progress = tqdm(
            total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            desc="Analyzing slides",
            unit="frames",
            disable=not self.config.debug,
        )

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if prev_frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(gray, prev_frame)
                    change_score = np.mean(diff)

                    if change_score > self.config.slide_threshold * 255:
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        if (not changes or 
                            (timestamp - changes[-1]) > self.config.min_slide_interval):
                            changes.append(timestamp)

                prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                progress.update(1)

            return changes
        finally:
            cap.release()
            progress.close()

    def _cleanup_temp_files(self, chunks: List[str]):
        """Clean up temporary files if configured"""
        if self.config.keep_files:
            return

        logger.info("Cleaning up temporary files...")
        for chunk in chunks:
            if os.path.exists(chunk):
                os.remove(chunk)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Lecture Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input parameters
    parser.add_argument("source", help="URL or file path of the lecture")
    parser.add_argument(
        "--platform",
        choices=["youtube", "coursera", "canvas", "local"],
        default="youtube",
        help="Source platform",
    )

    # Processing parameters
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="Maximum chunk size in MB"
    )
    parser.add_argument(
        "--chunk-duration", default="00:10:00", help="Duration per chunk (HH:MM:SS)"
    )

    # Audio parameters
    parser.add_argument(
        "--no-denoise",
        action="store_false",
        dest="denoise",
        help="Disable audio denoising",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Audio sample rate"
    )

    # Transcription parameters
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size",
    )
    parser.add_argument("--language", help="Force transcription language")
    parser.add_argument(
        "--beam-size", type=int, default=5, help="Whisper beam size for decoding"
    )

    # Analysis parameters
    parser.add_argument(
        "--text-window",
        type=int,
        default=3,
        help="Context window size for text analysis",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.4,
        help="Topic change detection threshold (0-1)",
    )
    parser.add_argument(
        "--slide-threshold",
        type=float,
        default=0.2,
        help="Slide change detection sensitivity (0-1)",
    )
    parser.add_argument(
        "--min-slide-interval",
        type=float,
        default=2.0,
        help="Minimum time between slide changes",
    )

    # Visualization parameters
    parser.add_argument(
        "--visualize", action="store_true", help="Generate analysis visualizations"
    )
    parser.add_argument(
        "--visualize-dir",
        default="visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="Display interactive plots"
    )

    # System parameters
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging and detailed outputs"
    )
    parser.add_argument(
        "--keep-files", action="store_true", help="Keep intermediate files"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed progress information"
    )

    return parser.parse_args()

def main():
    load_dotenv()  # Load environment variables (for HF_TOKEN)

    args = parse_args()
    config = AnalysisConfig(
        chunk_size=args.chunk_size,
        chunk_duration=args.chunk_duration,
        output_dir=args.output_dir,
        denoise_enabled=args.denoise,
        sample_rate=args.sample_rate,
        whisper_model=args.whisper_model,
        language=args.language,
        beam_size=args.beam_size,
        text_window_size=args.text_window,
        text_threshold=args.text_threshold,
        slide_threshold=args.slide_threshold,
        min_slide_interval=args.min_slide_interval,
        visualize=args.visualize,
        visualize_dir=args.visualize_dir,
        show_plots=args.show_plots,
        debug=args.debug,
        keep_files=args.keep_files,
    )

    # Configure logging level
    logger.setLevel(
        logging.DEBUG if args.debug else 
        logging.INFO if args.verbose else 
        logging.WARNING
    )

    try:
        analyzer = LectureAnalyzer(config)
        results = analyzer.process_lecture(args.source, args.platform)

        # Print summary
        logger.info("\nAnalysis Summary:")
        for result in results:
            print(f"\nChunk: {os.path.basename(result['chunk'])}")
            print(f"Detected {len(result['text_topics'])} topics")
            print(f"Found {len(result['slide_changes'])} slide changes")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
