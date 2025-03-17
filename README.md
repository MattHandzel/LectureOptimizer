# Lecture Optimizer

This software was made because of a realization that teachers are not created equal. They may be extremely knowledgeable about their field, but have not practiced the art of communicating and teaching effectively. This software assists the learner by taking in a video and outputting a higher quality video. 

### Features
- [ ] 30% faster lecture video *at the same speed*
    University lectures that are 1h20m long -> reduced to 50-55 minutes!
    With voice clonning, the lectures become ~50 minutes long, but they are much more understandable!
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

#### todo

- [ ] Replace letters with the phonetic equivalents of the letters (P -> pee)
- [ ] Add questions after lectures based on transcription to test understanding
- [ ] Smooth audio when using tts, don't need to remove silence because all audio is synthesized. My current system with voice clonning is actually very good. It is probably 70% of the way on par with normal lecture.
- [ ] I need to remove the awkward pauses that arise with silences being sped up
