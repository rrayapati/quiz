import os
import io
import math
import tempfile
from datetime import datetime
from typing import Tuple, List, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# Video + audio
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeAudioClip, concatenate_audioclips
import imageio_ffmpeg

# --- TTS options ---
# 1) OpenAI TTS (best for cloud if OPENAI_API_KEY is set)
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# 2) gTTS (works in cloud without system voices; requires internet)
GTTS_AVAILABLE = False
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# 3) pyttsx3 (offline on local machines with system voices; NOT reliable on Streamlit Cloud)
PYTTSX3_AVAILABLE = False
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False

# -----------------------------
# Utility: Fonts
# -----------------------------
def load_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Load a reasonable default font. We attempt DejaVuSans if present, otherwise fall back to PIL default.
    """
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

# -----------------------------
# TTS helpers (cloud-safe)
# -----------------------------
def tts_save_openai(text: str, out_path: str, voice: str = "alloy"):
    """
    Use OpenAI TTS to synthesize to MP3 (recommended on Streamlit Cloud).
    """
    if not OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI TTS unavailable: install openai and set OPENAI_API_KEY.")
    client = OpenAI()
    # Prefer a modern TTS model; fallback if necessary
    model_candidates = ["gpt-4o-mini-tts", "tts-1"]
    last_err = None
    for model in model_candidates:
        try:
            # Stream response to file if supported; else simple create
            try:
                with client.audio.speech.with_streaming_response.create(
                    model=model,
                    voice=voice,
                    input=text or ".",
                    format="mp3"
                ) as resp:
                    resp.stream_to_file(out_path)
                return
            except Exception:
                # older SDK path
                audio_resp = client.audio.speech.create(
                    model=model, voice=voice, input=text or ".", format="mp3"
                )
                with open(out_path, "wb") as f:
                    f.write(audio_resp.read())
                return
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"OpenAI TTS failed: {last_err}")

def tts_save_gtts(text: str, out_path: str):
    """
    Use gTTS to synthesize to MP3.
    """
    if not GTTS_AVAILABLE:
        raise RuntimeError("gTTS not installed.")
    tts = gTTS(text=text or ".", lang="en")
    tts.save(out_path)

def tts_save_pyttsx3(text: str, out_path: str, rate_delta: int = 0):
    """
    Use pyttsx3 to synthesize to WAV (local only; requires system voices like 'espeak' on Linux).
    """
    if not PYTTSX3_AVAILABLE:
        raise RuntimeError("pyttsx3 not installed.")
    engine = pyttsx3.init()
    rate = engine.getProperty("rate")
    engine.setProperty("rate", max(80, rate + rate_delta))
    engine.save_to_file(text or ".", out_path)
    engine.runAndWait()

def tts_save(text: str, out_path: str, mode: str, rate_delta: int = 0, voice: str = "alloy"):
    """
    Unified TTS wrapper.
    mode: "openai" | "gtts" | "pyttsx3"
    """
    if mode == "openai":
        return tts_save_openai(text, out_path, voice=voice)
    elif mode == "gtts":
        return tts_save_gtts(text, out_path)
    elif mode == "pyttsx3":
        return tts_save_pyttsx3(text, out_path, rate_delta=rate_delta)
    else:
        raise RuntimeError("Unknown TTS mode")

def make_audio_for_segments(segments: List[Tuple[str, str]], tmpdir: str, tts_mode: str, rate_delta: int = 0, voice: str = "alloy") -> Tuple[str, float]:
    """
    segments: list of (label, text)
    Returns (audio_path, total_duration_seconds)
    - For OpenAI/gTTS we create per-segment MP3s and concatenate.
    - For pyttsx3 we create WAVs and concatenate.
    """
    audio_paths = []
    ext = ".mp3" if tts_mode in ("openai", "gtts") else ".wav"
    for idx, (label, text) in enumerate(segments):
        out = os.path.join(tmpdir, f"seg_{idx:02d}_{label}{ext}")
        tts_save(text, out, mode=tts_mode, rate_delta=rate_delta, voice=voice)
        audio_paths.append(out)

    clips = [AudioFileClip(p) for p in audio_paths]
    full = concatenate_audioclips(clips)
    out_audio = os.path.join(tmpdir, f"narration{ext}")
    full.write_audiofile(out_audio, fps=44100, codec="aac" if ext==".mp3" else "pcm_s16le", verbose=False, logger=None)
    dur = full.duration
    try:
        for c in clips:
            c.close()
    except Exception:
        pass
    return out_audio, dur

# -----------------------------
# Layout helpers
# -----------------------------
def draw_safe_guides(img: Image.Image, bottom_reserved_ratio: float = 0.25, content_top_ratio: float = 0.08, content_bottom_ratio: float = 0.70) -> Image.Image:
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    reserved_h = int(h * bottom_reserved_ratio)
    draw.rectangle([0, h - reserved_h, w, h], fill=(255, 0, 0, 40))
    y1 = int(h * content_top_ratio)
    y2 = int(h * content_bottom_ratio)
    draw.rectangle([int(w*0.06), y1, int(w*0.94), y2], outline=(0, 255, 0, 160), width=4)
    return img

def paste_centered(bg: Image.Image, overlay: Image.Image, box: Tuple[int, int, int, int]):
    l, t, r, b = box
    bw, bh = r - l, b - t
    ow, oh = overlay.size
    scale = min(bw / ow, bh / oh)
    new_size = (max(1, int(ow * scale)), max(1, int(oh * scale)))
    overlay_resized = overlay.resize(new_size, Image.LANCZOS)
    ox = l + (bw - new_size[0]) // 2
    oy = t + (bh - new_size[1]) // 2
    bg.paste(overlay_resized, (ox, oy), overlay_resized if overlay_resized.mode == "RGBA" else None)
    return bg

def text_wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    lines = []
    if not words:
        return [""]
    cur = words[0]
    for w in words[1:]:
        if draw.textlength(cur + " " + w, font=font) <= max_width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines

def draw_text_block(img: Image.Image, text: str, font: ImageFont.ImageFont, xy: Tuple[int, int], fill=(255,255,255), max_width: Optional[int]=None, line_spacing_ratio: float=1.15) -> Tuple[int,int]:
    draw = ImageDraw.Draw(img)
    if max_width is None:
        draw.text(xy, text, font=font, fill=fill)
        bbox = draw.textbbox(xy, text, font=font)
        return bbox[2]-bbox[0], bbox[3]-bbox[1]
    lines = text_wrap(draw, text, font, max_width)
    x, y = xy
    line_h = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
    total_h = 0
    max_w = 0
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        w = draw.textlength(line, font=font)
        max_w = max(max_w, int(w))
        y += int(line_h * line_spacing_ratio)
        total_h += int(line_h * line_spacing_ratio)
    return max_w, total_h

# -----------------------------
# Frame building
# -----------------------------
def base_canvas(bg_image: Optional[Image.Image], size=(1080, 1920)) -> Image.Image:
    canvas = Image.new("RGB", size, (15, 15, 20))
    if bg_image:
        bg = bg_image.copy().convert("RGB").resize(size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=1.2))
        overlay = Image.new("RGBA", size, (0,0,0,110))
        bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
        canvas.paste(bg, (0,0))
    return canvas

def render_title_frame(bg_img: Optional[Image.Image], title: str, day: str, quiz_no: str, size=(1080,1920), show_guides=False) -> Image.Image:
    img = base_canvas(bg_img, size=size)
    w, h = img.size
    title_font = load_font(80)
    meta_font = load_font(46)

    margin = int(w*0.06)
    top_box = (margin, int(h*0.08), w-margin, int(h*0.24))
    draw = ImageDraw.Draw(img)
    draw_text_block(img, title, title_font, (top_box[0], top_box[1]), fill=(255,255,255), max_width=(top_box[2]-top_box[0]))
    meta_text = f"{day} • {quiz_no}"
    draw_text_block(img, meta_text, meta_font, (top_box[0], top_box[1]+int((top_box[3]-top_box[1])*0.55)), fill=(220,220,220), max_width=(top_box[2]-top_box[0]))

    if show_guides:
        img = draw_safe_guides(img)
    return img

def render_question_frame(bg_img: Optional[Image.Image], question_partial: str, size=(1080,1920), show_guides=False) -> Image.Image:
    img = base_canvas(bg_img, size=size)
    w, h = img.size
    if show_guides:
        img = draw_safe_guides(img)
    content_box = (int(w*0.06), int(h*0.20), int(w*0.94), int(h*0.70))
    q_font = load_font(56)
    draw_text_block(img, question_partial, q_font, (content_box[0], content_box[1]), fill=(255,255,255), max_width=(content_box[2]-content_box[0]))
    return img

def render_options_frame(bg_img: Optional[Image.Image], question: str, options: List[str], alphas: List[float], size=(1080,1920), show_guides=False) -> Image.Image:
    img = base_canvas(bg_img, size=size)
    w, h = img.size
    if show_guides:
        img = draw_safe_guides(img)
    content_box = (int(w*0.06), int(h*0.20), int(w*0.94), int(h*0.70))
    q_font = load_font(54)
    opt_font = load_font(48)
    draw_text_block(img, question, q_font, (content_box[0], content_box[1]), fill=(255,255,255), max_width=(content_box[2]-content_box[0]))
    draw = ImageDraw.Draw(img, "RGBA")
    opt_y = int(h*0.36)
    opt_gap = int(h*0.06)
    labels = ["A", "B", "C", "D"]
    for i, opt in enumerate(options):
        alpha = max(0.0, min(1.0, alphas[i] if i < len(alphas) else 1.0))
        fill = (255,255,255, int(255*alpha))
        label_fill = (180,220,255, int(255*alpha))
        draw.rounded_rectangle([content_box[0], opt_y-10, content_box[0]+48, opt_y+48], radius=10, fill=label_fill)
        draw_text_block(img, labels[i], opt_font, (content_box[0]+10, opt_y-2), fill=(20,30,40), max_width=None)
        draw.text((content_box[0]+64, opt_y), opt, font=opt_font, fill=fill)
        opt_y += opt_gap
    return img

def render_answer_frame(bg_img: Optional[Image.Image], question: str, options: List[str], correct_idx: int, explanation: str, size=(1080,1920), show_guides=False) -> Image.Image:
    img = base_canvas(bg_img, size=size)
    w, h = img.size
    if show_guides:
        img = draw_safe_guides(img)
    content_box = (int(w*0.06), int(h*0.20), int(w*0.94), int(h*0.70))
    q_font = load_font(54)
    opt_font = load_font(48)
    exp_font = load_font(42)

    draw_text_block(img, question, q_font, (content_box[0], content_box[1]), fill=(255,255,255), max_width=(content_box[2]-content_box[0]))

    draw = ImageDraw.Draw(img, "RGBA")
    opt_y = int(h*0.36)
    opt_gap = int(h*0.06)
    labels = ["A", "B", "C", "D"]
    for i, opt in enumerate(options):
        is_correct = (i == correct_idx)
        bg_col = (80, 200, 120, 220) if is_correct else (255,255,255,50)
        txt_col = (10,25,15) if is_correct else (255,255,255)
        draw.rounded_rectangle([content_box[0], opt_y-16, content_box[2], opt_y+56], radius=16, fill=bg_col)
        draw.rounded_rectangle([content_box[0]+8, opt_y-6, content_box[0]+58, opt_y+44], radius=10, fill=(255,255,255,230) if is_correct else (180,220,255,160))
        draw_text_block(img, labels[i], opt_font, (content_box[0]+16, opt_y-2), fill=(10,25,15) if is_correct else (20,30,40))
        draw.text((content_box[0]+74, opt_y), opt, font=opt_font, fill=txt_col)
        opt_y += opt_gap

    exp_box_top = opt_y + int(h*0.02)
    exp_area = (content_box[0], exp_box_top, content_box[2], int(h*0.70))
    exp_text = f"Why: {explanation}"
    draw_text_block(img, exp_text, exp_font, (exp_area[0], exp_area[1]), fill=(230,230,230), max_width=(exp_area[2]-exp_area[0]))
    return img

def render_engagement_frame(bg_img: Optional[Image.Image], size=(1080,1920), show_guides=False) -> Image.Image:
    img = base_canvas(bg_img, size=size)
    if show_guides:
        img = draw_safe_guides(img)
    w, h = img.size
    big = load_font(64)
    small = load_font(44)
    draw = ImageDraw.Draw(img)
    draw_text_block(img, "COMMENT YOUR ANSWER BELOW!", big, (int(w*0.08), int(h*0.28)), fill=(255,255,255), max_width=int(w*0.84))
    draw_text_block(img, "We’ll reveal more in the next video.", small, (int(w*0.08), int(h*0.38)), fill=(220,220,220), max_width=int(w*0.84))
    return img

def render_outro_frame(bg_img: Optional[Image.Image], size=(1080,1920), show_guides=False) -> Image.Image:
    img = base_canvas(bg_img, size=size)
    if show_guides:
        img = draw_safe_guides(img)
    w, h = img.size
    big = load_font(64)
    small = load_font(44)
    draw = ImageDraw.Draw(img)
    draw_text_block(img, "LIKE • SHARE • SUBSCRIBE", big, (int(w*0.08), int(h*0.30)), fill=(255,255,255), max_width=int(w*0.84))
    draw_text_block(img, "New quizzes daily! 🔔", small, (int(w*0.08), int(h*0.40)), fill=(220,220,220), max_width=int(w*0.84))
    return img

# -----------------------------
# OpenAI question generation (optional)
# -----------------------------
def generate_quiz_via_openai(topic: str, difficulty: str) -> Tuple[str, List[str], int, str]:
    if not OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI not available. Set OPENAI_API_KEY and install openai>=1.0.")
    client = OpenAI()
    sys_prompt = "You are a quiz writer. Return JSON with keys: question, options (4 items), correct_index (0-3), explanation (1-2 sentences)."
    user_prompt = f"Create one multiple-choice question about '{topic}' at {difficulty} difficulty. Do not add extra keys."
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":sys_prompt},
            {"role":"user","content":user_prompt}
        ],
        temperature=0.7,
    )
    content = resp.choices[0].message.content
    import json, re
    try:
        j = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            raise RuntimeError("OpenAI response not JSON.")
        j = json.loads(m.group())
    q = j["question"].strip()
    opts = [o.strip() for o in j["options"]][:4]
    ci = int(j["correct_index"])
    expl = j["explanation"].strip()
    if len(opts) != 4:
        raise RuntimeError("OpenAI did not return 4 options.")
    return q, opts, ci, expl

# -----------------------------
# Video assembly
# -----------------------------
def assemble_video(frames: List[Image.Image], audio_path: Optional[str], fps: int, out_path: str):
    arr_frames = [np.array(im.convert("RGB")) for im in frames]
    clip = ImageSequenceClip(arr_frames, fps=fps)
    if audio_path and os.path.exists(audio_path):
        audio = AudioFileClip(audio_path)
        clip = clip.set_audio(audio)
    clip.write_videofile(out_path, fps=fps, codec="libx264", audio_codec="aac", preset="medium", threads=2, verbose=False, logger=None)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🎬 Quiz Video Generator", page_icon="❓", layout="centered")
st.title("🎬 YouTube Quiz Video Generator (MVP)")

with st.expander("📋 Title & Meta", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        quiz_title = st.text_input("Quiz Title", value="GENERAL KNOWLEDGE QUIZ")
        day_str = st.text_input("Day Label", value="Day 1")
    with col2:
        quiz_no = st.text_input("Quiz Number Label", value="Quiz #1")
        resolution = st.selectbox("Resolution", options=["1080x1920 (Vertical 9:16)", "720x1280 (Vertical 9:16)"], index=0)
    bg_file = st.file_uploader("Background Image (optional)", type=["png","jpg","jpeg","webp"])
    show_guides = st.checkbox("Show safe-zone guides (bottom 25% reserved)", value=True)

with st.expander("🧠 Question Source", expanded=True):
    auto_generate = st.checkbox("Auto-generate using OpenAI", value=False, help="Requires OPENAI_API_KEY in env and 'openai' SDK installed.")
    topic = st.text_input("Topic (for OpenAI)", value="World Geography")
    difficulty = st.selectbox("Difficulty", ["easy","medium","hard"], index=1)
    st.caption("Or fill manually below:")

with st.expander("✍️ Question & Options", expanded=not auto_generate):
    question = st.text_area("Question", value="Which country has the largest land area in the world?")
    optA = st.text_input("Option A", value="Russia")
    optB = st.text_input("Option B", value="Canada")
    optC = st.text_input("Option C", value="China")
    optD = st.text_input("Option D", value="United States")
    correct_idx = st.selectbox("Correct Option", options=[0,1,2,3], format_func=lambda i: ["A","B","C","D"][i], index=0)
    explanation = st.text_area("Answer Explanation (1–2 lines)", value="Russia spans Eastern Europe and northern Asia, making it the world's largest country by area.")

with st.expander("🔊 Voice-over (choose mode)", expanded=True):
    tts_mode = st.selectbox("TTS Mode", options=["openai (cloud)", "gtts (cloud)", "pyttsx3 (local)"], index=0)
    voice = st.text_input("Voice (OpenAI only)", value="alloy", help="For OpenAI TTS models.")
    rate_delta = st.slider("Local voice speed (pyttsx3 only)", min_value=-60, max_value=60, value=-10, step=5)

with st.expander("⚙️ Timing & Effects", expanded=True):
    fps = st.slider("FPS", 20, 40, 30, 1)
    chars_per_frame = st.slider("Typing speed (chars per frame)", 1, 5, 2, 1)
    option_fade_frames = st.slider("Fade frames per option", 5, 30, 12, 1)
    hold_title = st.slider("Hold Title (seconds)", 0, 5, 2, 1)
    hold_after_options = st.slider("Hold After Options (seconds)", 0, 5, 1, 1)
    hold_answer = st.slider("Hold Answer & Explanation (seconds)", 1, 8, 4, 1)
    hold_engage = st.slider("Hold 'Comment your answer' (seconds)", 1, 8, 3, 1)
    hold_outro = st.slider("Hold Outro (seconds)", 1, 8, 3, 1)

st.markdown("---")
generate = st.button("🎥 Generate Video")

if generate:
    try:
        if resolution.startswith("1080"):
            size = (1080, 1920)
        else:
            size = (720, 1280)

        bg_img = None
        if bg_file is not None:
            bg_img = Image.open(bg_file)

        if auto_generate:
            q, opts, ci, expl = generate_quiz_via_openai(topic, difficulty)
            question = q
            optA, optB, optC, optD = opts
            correct_idx = ci
            explanation = expl

        options = [optA, optB, optC, optD]

        # Prepare audio narration segments
        segments = []
        segments.append(("title", f"{quiz_title}. {day_str}. {quiz_no}."))
        segments.append(("question", question))
        segments.append(("options", f"Option A: {optA}. Option B: {optB}. Option C: {optC}. Option D: {optD}."))
        answer_letter = ["A","B","C","D"][correct_idx]
        segments.append(("answer", f"The correct answer is {answer_letter}. {explanation}"))
        segments.append(("engage", "Now your turn. Comment your answer below."))
        segments.append(("outro", "Please like, share, and subscribe for more quiz videos."))

        # Choose TTS mode
        selected_mode = tts_mode.split()[0]  # "openai" | "gtts" | "pyttsx3"

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path, total_audio = make_audio_for_segments(
                segments, tmpdir, tts_mode=selected_mode, rate_delta=rate_delta, voice=voice
            )

            frames: List[Image.Image] = []

            title_frame = render_title_frame(bg_img, quiz_title, day_str, quiz_no, size=size, show_guides=show_guides)
            frames += [title_frame] * max(1, hold_title * fps)

            text = question or "."
            for i in range(1, len(text)+1, chars_per_frame):
                partial = text[:i]
                frames.append(render_question_frame(bg_img, partial, size=size, show_guides=show_guides))

            alphas = [0.0, 0.0, 0.0, 0.0]
            for idx in range(4):
                for f in range(option_fade_frames):
                    a = min(1.0, (f+1)/option_fade_frames)
                    cur = alphas.copy()
                    cur[idx] = a
                    frames.append(render_options_frame(bg_img, question, options, cur, size=size, show_guides=show_guides))

            frames += [render_options_frame(bg_img, question, options, [1.0,1.0,1.0,1.0], size=size, show_guides=show_guides)] * max(1, hold_after_options * fps)

            ans_frame = render_answer_frame(bg_img, question, options, correct_idx, explanation, size=size, show_guides=show_guides)
            frames += [ans_frame] * max(1, hold_answer * fps)

            engage = render_engagement_frame(bg_img, size=size, show_guides=show_guides)
            frames += [engage] * max(1, hold_engage * fps)

            outro = render_outro_frame(bg_img, size=size, show_guides=show_guides)
            frames += [outro] * max(1, hold_outro * fps)

            out_name = f"quiz_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            out_path = os.path.join(tmpdir, out_name)
            arr_frames = [np.array(im.convert("RGB")) for im in frames]
            clip = ImageSequenceClip(arr_frames, fps=fps)
            if audio_path and os.path.exists(audio_path):
                audio = AudioFileClip(audio_path)
                clip = clip.set_audio(audio)
            clip.write_videofile(out_path, fps=fps, codec="libx264", audio_codec="aac", preset="medium", threads=2, verbose=False, logger=None)

            with open(out_path, "rb") as f:
                data = f.read()
            st.success("Video generated!")
            st.download_button("⬇️ Download MP4", data=data, file_name=out_name, mime="video/mp4")

    except Exception as e:
        st.error(f"TTS/Video error: {e}")
        st.exception(e)
