# 🎬 YouTube Quiz Video Generator (Streamlit MVP)

Generate vertical (9:16) **quiz videos** with:
- Title (includes **Day** + **Quiz #**)
- **Background image**
- **Typing effect** question
- 4 options with **fade-in**
- **Answer highlight** + short explanation
- **"Comment your answer"** engagement slide
- **Outro**: “Please Like, Share & Subscribe”
- **AI voice-over** (offline via `pyttsx3`)
- Optional **OpenAI** question generation

https://github.com/  ← (optional Git init; not required)

---

## ✨ Features
- **Safe zones**: bottom **25%** reserved for YouTube UI (toggle guides on/off).
- **Vertical resolutions**: 1080×1920 or 720×1280.
- **No external video editor** needed—renders an MP4 directly.
- **Works offline** for TTS using system voices (`pyttsx3`).

> Note: On Linux you may need to install `espeak` for `pyttsx3`:
> ```bash
> sudo apt-get update && sudo apt-get install -y espeak ffmpeg
> ```

---

## 🚀 Quickstart

### 1) Create & activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> If `moviepy` complains about `ffmpeg`, install it:
> - macOS (brew): `brew install ffmpeg`
> - Ubuntu/Debian: `sudo apt-get install -y ffmpeg`
> - Windows: install FFmpeg and add it to PATH, or use `imageio-ffmpeg` (already included).

### 3) (Optional) Enable OpenAI question generation
Set your API key:
```bash
export OPENAI_API_KEY=sk-...   # Windows PowerShell: setx OPENAI_API_KEY "sk-..."
```
This allows you to auto-generate a question, 4 options, the correct index, and a short explanation.

### 4) Run the app
```bash
streamlit run app.py
```
Open the local URL shown by Streamlit in your browser.

---

## 🧠 How it works

- **Frames** are rendered with **Pillow** (no ImageMagick required).
- **Typing effect**: generates incremental frames showing the question progressively.
- **Options**: each of the four options **fades in** sequentially.
- **Answer**: correct option is highlighted with a green pill; an explanation appears below.
- **Engagement** + **Outro** frames follow.
- **Audio**: Text-to-speech is generated for segments (title → question → options → answer → engagement → outro) and concatenated into a single WAV, then muxed into the MP4 via MoviePy.

> 🔊 If you want different voices:
> - On **Windows**, open “Voice settings” (SAPI5) and install additional voices.
> - On **macOS**, add voices in **System Settings → Accessibility → Spoken Content**.
> - On **Linux**, install additional `espeak` voices.

---

## 🧩 Customization

Inside the app:
- **Typing speed** (`chars_per_frame`)
- **Fade speed** (`option_fade_frames`)
- **Segment holds** (title, after options, answer, engagement, outro)
- **FPS** (20–40)
- **Show/Hide safe-zone guides**

Design:
- You can adjust colors, fonts, and layout in `app.py` (search for `render_*_frame` functions).

---

## 📦 Outputs

- Produces an **MP4** (H.264 + AAC) suitable for YouTube Shorts/Reels.
- Download it directly from the Streamlit app.

---

## ❓ FAQ

**Q: The voice sounds too fast/slow.**  
A: Use the **Voice Speed Adjustment** slider (in the app).

**Q: MoviePy says FFmpeg not found.**  
A: Install FFmpeg or ensure PATH is set. See Quickstart.

**Q: Can I use OpenAI voices instead?**  
A: Yes—replace `pyttsx3` with OpenAI’s TTS in `app.py` (commented area is provided). This requires internet and an API key.

**Q: Can I change fonts to Montserrat?**  
A: Install fonts on your OS and point `load_font()` to the `.ttf` (e.g., `Montserrat-ExtraBold.ttf`).

---

## ⚠️ Notes

- This MVP keeps **all main content** within the **upper 70%** of the canvas and **reserves the bottom 25%** for YouTube UI overlays.
- If your question is extremely long, reduce the font size or shorten the text to keep it inside the safe zone.
- On some Linux servers, `pyttsx3` requires `espeak` + audio backends to be present.

---

## 📄 License
MIT (modify and use freely for your channel).

Enjoy building! 🎉
