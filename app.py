import os
import tempfile
from datetime import datetime
from typing import Tuple, List, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_audioclips

# gTTS for cloud-safe narration (toggleable)
GTTS_AVAILABLE = False
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

APP_BUILD = "manual-only-2025-09-10T18:10+05:30"

# -----------------------------
# Utilities
# -----------------------------
def load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def text_wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = (text or "").split()
    if not words:
        return [""]
    lines, cur = [], words[0]
    for w in words[1:]:
        if draw.textlength(cur + " " + w, font=font) <= max_width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines

def draw_text_block(img: Image.Image, text: str, font: ImageFont.ImageFont, xy: Tuple[int, int], fill=(255,255,255), max_width: Optional[int]=None, line_spacing=1.15) -> Tuple[int,int]:
    draw = ImageDraw.Draw(img)
    if max_width is None:
        draw.text(xy, text or "", font=font, fill=fill)
        bbox = draw.textbbox(xy, text or "", font=font)
        return bbox[2]-bbox[0], bbox[3]-bbox[1]
    lines = text_wrap(draw, text or "", font, max_width)
    x, y = xy
    line_h = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
    total_h, max_w = 0, 0
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        w = draw.textlength(line, font=font)
        max_w = max(max_w, int(w))
        y += int(line_h * line_spacing)
        total_h += int(line_h * line_spacing)
    return max_w, total_h

def draw_safe_guides(img: Image.Image, bottom_reserved_ratio: float = 0.25, content_top_ratio: float = 0.08, content_bottom_ratio: float = 0.70) -> Image.Image:
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    reserved_h = int(h * bottom_reserved_ratio)
    draw.rectangle([0, h - reserved_h, w, h], fill=(255, 0, 0, 40))
    y1 = int(h * content_top_ratio)
    y2 = int(h * content_bottom_ratio)
    draw.rectangle([int(w*0.06), y1, int(w*0.94), y2], outline=(0, 255, 0, 160), width=4)
    return img

# -----------------------------
# TTS (gTTS only, optional)
# -----------------------------
def tts_save_gtts(text: str, out_path: str):
    if not GTTS_AVAILABLE:
        raise RuntimeError("gTTS not installed")
    tts = gTTS(text=(text or "."), lang="en")
    tts.save(out_path)

def make_audio_for_segments(segments: List[Tuple[str, str]], tmpdir: str) -> Tuple[Optional[str], float]:
    """Return (audio_path, duration). If narration disabled or fails, returns (None, 0)."""
    try:
        ext = ".mp3"
        audio_paths = []
        for idx, (label, text) in enumerate(segments):
            path = os.path.join(tmpdir, f"seg_{idx:02d}_{label}{ext}")
            tts_save_gtts(text, path)
            audio_paths.append(path)
        clips = [AudioFileClip(p) for p in audio_paths]
        full = concatenate_audioclips(clips)
        out_audio = os.path.join(tmpdir, f"narration{ext}")
        # Let MoviePy infer codec from extension
        full.write_audiofile(out_audio, fps=44100, verbose=False, logger=None)
        dur = full.duration
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
        return out_audio, dur
    except Exception:
        # If any audio error occurs, fall back to silent video
        return None, 0.0

# -----------------------------
# Frame builders
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
        draw_text_block(img, labels[i], opt_font, (content_box[0]+10, opt_y-2), fill=(20,30,40))
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
    exp_area_w = int(w*0.88)
    exp_x = int(w*0.06)
    draw_text_block(img, f"Why: {explanation}", exp_font, (exp_x, exp_box_top), fill=(230,230,230), max_width=exp_area_w)
    return img

def render_engagement_frame(bg_img: Optional[Image.Image], size=(1080,1920), show_guides=False) -> Image.Image:
    img = base_canvas(bg_img, size=size)
    if show_guides:
        img = draw_safe_guides(img)
    w, h = img.size
    big = load_font(64)
    small = load_font(44)
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
    draw_text_block(img, "LIKE • SHARE • SUBSCRIBE", big, (int(w*0.08), int(h*0.30)), fill=(255,255,255), max_width=int(w*0.84))
    draw_text_block(img, "New quizzes daily! 🔔", small, (int(w*0.08), int(h*0.40)), fill=(220,220,220), max_width=int(w*0.84))
    return img

# -----------------------------
# Streamlit UI (Manual-only)
# -----------------------------
st.set_page_config(page_title="🎬 Quiz Video Generator (Manual Only)", page_icon="❓", layout="centered")
st.title("🎬 YouTube Quiz Video Generator — Manual Only")

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

with st.expander("✍️ Question & Options", expanded=True):
    question = st.text_area("Question", value="Which country has the largest land area in the world?")
    optA = st.text_input("Option A", value="Russia")
    optB = st.text_input("Option B", value="Canada")
    optC = st.text_input("Option C", value="China")
    optD = st.text_input("Option D", value="United States")
    correct_idx = st.selectbox("Correct Option", options=[0,1,2,3], format_func=lambda i: ["A","B","C","D"][i], index=0)
    explanation = st.text_area("Answer Explanation (1–2 lines)", value="Russia spans Eastern Europe and northern Asia, making it the world's largest country by area.")

with st.expander("🔊 Narration (gTTS)", expanded=True):
    enable_tts = st.checkbox("Enable narration (gTTS)", value=True)
    st.caption("Cloud-friendly TTS. If it fails, the app will still render a silent video.")

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
with st.expander("🔧 Diagnostics", expanded=False):
    st.write(f"Build: **{APP_BUILD}**")
    st.write(f"gTTS available: **{GTTS_AVAILABLE}**")

generate = st.button("🎥 Generate Video")

if generate:
    try:
        size = (1080, 1920) if resolution.startswith("1080") else (720, 1280)

        bg_img = None
        if bg_file is not None:
            bg_img = Image.open(bg_file)

        options = [optA, optB, optC, optD]

        segments = [
            ("title", f"{quiz_title}. {day_str}. {quiz_no}."),
            ("question", question),
            ("options", f"Option A: {optA}. Option B: {optB}. Option C: {optC}. Option D: {optD}."),
            ("answer", f"The correct answer is {['A','B','C','D'][correct_idx]}. {explanation}"),
            ("engage", "Now your turn. Comment your answer below."),
            ("outro", "Please like, share, and subscribe for more quiz videos."),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Audio (optional)
            audio_path = None
            if enable_tts and GTTS_AVAILABLE:
                audio_path, _ = make_audio_for_segments(segments, tmpdir)
                if audio_path is None:
                    st.warning("Narration failed. Rendering a silent video instead.")

            frames: List[Image.Image] = []

            # Title
            title_frame = render_title_frame(bg_img, quiz_title, day_str, quiz_no, size=size, show_guides=show_guides)
            frames += [title_frame] * max(1, hold_title * fps)

            # Typing question
            txt = question or "."
            for i in range(1, len(txt)+1, chars_per_frame):
                frames.append(render_question_frame(bg_img, txt[:i], size=size, show_guides=show_guides))

            # Options fade-in
            alphas = [0.0, 0.0, 0.0, 0.0]
            for idx in range(4):
                for f in range(option_fade_frames):
                    a = min(1.0, (f+1)/option_fade_frames)
                    cur = alphas.copy()
                    cur[idx] = a
                    frames.append(render_options_frame(bg_img, question, options, cur, size=size, show_guides=show_guides))

            frames += [render_options_frame(bg_img, question, options, [1,1,1,1], size=size, show_guides=show_guides)] * max(1, hold_after_options * fps)

            # Answer + explanation
            ans = render_answer_frame(bg_img, question, options, correct_idx, explanation, size=size, show_guides=show_guides)
            frames += [ans] * max(1, hold_answer * fps)

            # Engagement
            engage = render_engagement_frame(bg_img, size=size, show_guides=show_guides)
            frames += [engage] * max(1, hold_engage * fps)

            # Outro
            outro = render_outro_frame(bg_img, size=size, show_guides=show_guides)
            frames += [outro] * max(1, hold_outro * fps)

            out_name = f"quiz_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            out_path = os.path.join(tmpdir, out_name)
            clip = ImageSequenceClip([np.array(im.convert('RGB')) for im in frames], fps=fps)
            if audio_path and os.path.exists(audio_path):
                clip = clip.set_audio(AudioFileClip(audio_path))
            clip.write_videofile(out_path, fps=fps, codec="libx264", audio_codec="aac", preset="medium", threads=2, verbose=False, logger=None)

            with open(out_path, "rb") as f:
                data = f.read()
            st.success("Video generated!")
            st.download_button("⬇️ Download MP4", data=data, file_name=out_name, mime="video/mp4")

    except Exception as e:
        st.error(f"TTS/Video error: {e}")
        st.exception(e)
