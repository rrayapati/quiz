# ☁️ Streamlit Cloud–Friendly Quiz Video Generator

This version adds **cloud-safe TTS** so it runs on **Streamlit Cloud** without system voices.

## What changed
- New **TTS Mode** selector:
  - **openai (cloud)** – recommended (set `OPENAI_API_KEY`)
  - **gtts (cloud)** – no API key; uses Google TTS
  - **pyttsx3 (local)** – for laptops/desktops with system voices
- Audio segments are generated as MP3 for cloud modes and concatenated in-app.
- Same visuals: title (with Day + Quiz #), background image, typing question, 4 option fade-in, answer + explanation, **"Comment your answer"**, and **Outro**.

---

## Deploy on Streamlit Cloud

1. Push `app.py`, `requirements.txt`, `README.md` to GitHub.
2. On Streamlit Cloud, set **secrets / environment variables**:
   - For **OpenAI TTS** (recommended):
     - `OPENAI_API_KEY = sk-...`
3. Deploy. In the app, choose **TTS Mode → openai (cloud)** (or **gtts (cloud)**).

> If you select `pyttsx3 (local)` on Streamlit Cloud, it will fail due to missing `espeak`. Use **openai** or **gtts** instead.

---

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

- On local you can use **pyttsx3** if your OS has voices installed.
- Or use **openai**/**gtts** like in the cloud.

---

## Notes
- OpenAI TTS models: tries `gpt-4o-mini-tts`, then `tts-1` if needed.
- gTTS requires internet access but no API key.
- Output MP4 uses **H.264 + AAC**, ready for YouTube Shorts.

Enjoy! 🎉
