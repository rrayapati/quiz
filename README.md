# 🎬 Quiz Video Generator — Manual Only (Streamlit)

Create vertical (9:16) quiz videos with:
- Title (with **Day** + **Quiz #**)
- Optional **background image**
- **Typing-effect** question
- 4 options with **sequential fade-in**
- **Answer highlight** + short **explanation**
- **"Comment your answer"** slide
- **Outro** slide
- Optional **narration** using **gTTS** (cloud-friendly)

---

## Quickstart

### Local
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud
1. Push `app.py`, `requirements.txt`, `README.md` to GitHub.
2. Deploy on Streamlit Cloud (no secrets needed).
3. If narration fails due to outbound network, uncheck **“Enable narration (gTTS)”** and render a silent video.

---

## Controls
- **Resolution**: 1080×1920 (default) or 720×1280 (vertical).
- **Safe zones**: bottom **25%** reserved (toggle guide), main content within **8%–70%** of height.
- **Timing**: FPS, typing speed, fade duration, hold durations for each segment.
- **Narration**: Toggle on/off. Uses gTTS to generate per-segment MP3, concatenates them, and muxes into MP4.

---

## Troubleshooting
- If you see audio-related errors, toggle **Narration** off and render a silent video first.
- For **FFmpeg** issues locally, install it:
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install -y ffmpeg`
  - Windows: install FFmpeg and add it to PATH.
- We **don’t** pass explicit codecs when writing audio; MoviePy infers them from the file extension to avoid the known `'FFMPEG_AudioWriter' object has no attribute 'ext'` bug.

---

## License
MIT
