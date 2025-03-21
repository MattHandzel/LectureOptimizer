# Lecture Optimizer

This software was made for people who are watching unedited, unpolished educational material and want to improve the quality of the lecture. This software assists in learning through the philosophy of removing what does not give the student value.

### Example

##### MIT Linear Algebra

**Original (45m55s)**:

<video src="https://www.youtube.com/watch?v=2IdtqGM6KWU" controls="controls" style="max-width: 730px;"></video>

[![Original Lecture 11 Video](https://img.youtube.com/vi/2IdtqGM6KWU/0.jpg)](https://www.youtube.com/watch?v=2IdtqGM6KWU)

**Optimized (35m19s)**:

[![Original Lecture 11 Video](https://img.youtube.com/vi/EtxPOWO6bgc/0.jpg)](https://youtu.be/EtxPOWO6bgc)

My tool speeds up this lecture by ~23\%


#### Voice Cloning 🤯

Using [OpenVoice](https://github.com/myshell-ai/OpenVoice), it is possible to increase the audio quality and speed through voice cloning. Here is an example taking the first 10 minutes of the above lecture:

**Clonned (5m02s)**:


https://github.com/user-attachments/assets/bfd4522c-4f17-4f00-b3a6-c248ff60ad91

Notice that the speed gain by voice cloning is much more than the original tool. The trade-off is that, on my machine (which has no GPU), for every minute of video that uses voice cloning, it takes around 3-4 minutes to generate (most of it being the text to speed), so it cannot be done in real time 😭.

>  WARNING: Do not use the voice cloning tool on anyone without their permission. This is an example of voice cloning using my own voice.

### Features

- [x] Analyzes the lecture video and remove/speed up the silent parts inspired by the [jumpcutter](https://github.com/carykh/jumpcutter)
- [x] Improves audio quality by denoising the audio
- [ ] Pass in a link to download and process the video
- [x] Capability of transcribing the lecture to copy and paste into notes
- [x] Implementing voice-clonning to reduce accents, standardize speaker cadence, improving understanding
- [ ] Extract text from slides
- [ ] Removing filler words, rewording the lecture to improve understanding
- [ ] Integrating with the user's notes to skip past material that is already understood
- [ ] Using learning theory and active learning to improve the lecture quality (inserting questions, etc.)
- [ ] Multilingual support
- [ ] Browser integration

### Installation

Clone this repo: `git@github.com:MattHandzel/LectureOptimizer.git`

#### Voice Cloning Installation

Clone the repo and follow the instructions of: `https://github.com/myshell-ai/MeloTTS` and install the dependencies with `pip install -r MeloTTS/requirements.txt`

### Usage

The `optimize_lecture.py` script is the main entry point for the software. It takes in a video file and processes it to an output directory.

Example:

```sh
python optimize_lecture.py ./input/linear-algebra-lecture.mp4  --output_dir ./output/linear-algebra-lecture/ --speed_up --fps 30 --num_workers 15 --normal_speed 1 --silent_speed 10 --padding 250 --silence_threshold -48 --min_silence_len 750
```

This will take in the input lecture at `./input/linear-algebra-lecture.mp4` and output the processed lecture to `./output/linear-algebra-lecture/`. The video will be sped up so that the silent sections will be sped up by a factor of 10, and the non-silent sections will be sped up by a factor of 1. The video will be processed at 30 fps. When a silent section is detected, a padding of is 250ms will be used so consonants are not cut-off and the audio sounds better. The minimum silence length is 750ms and the silence threshold is -48db.

#### Voice Cloning

Here is an example of voice clonning:

```sh
python optimize_lecture.py ./input/linear-algebra-lecture-short.mp4  --output_dir ./output/linear-algebra-lecture-short-voice-clone/ --speed_up --fps 30 --num_workers 15 --normal_speed 1 --silent_speed 10 --padding 250 --silence_threshold -52 --min_silence_len 750 --clone_voice --source_voice ./matts-voice.mp3 --transcript_output ./output/linear-algebra-lecture-short-voice-clone/linear-algebra-short.tsv
```

#### todo

- [ ] Replace letters with the phonetic equivalents of the letters (P -> pee)
- [ ] Add questions after lectures based on transcription to test understanding
- [ ] Smooth audio when using tts, don't need to remove silence because all audio is synthesized. My current system with voice clonning is actually very good. It is probably 70% of the way on par with normal lecture.
- [ ] I need to remove the awkward pauses that arise with silences being sped up
- [ ] Use OpenVoice V2 for voice clonning
