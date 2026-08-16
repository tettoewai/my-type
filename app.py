"""MyType - main Streamlit application entry point."""

from __future__ import annotations

import base64

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
import mediapipe as mp

from modules import face_analyzer, skin_analyzer, body_analyzer, recommender
from utils.helpers import load_image, resize_to_max_dimension

ICON = lambda name: f'<i class="fa-solid fa-{name}"></i>'

MAX_IMAGE_DIMENSION = 1024
HEAD_CROP_SIZE = 512
POSE_HEAD_LANDMARKS = [0, 7, 8, 2]

BODY_SHAPES_FEMALE = ["Inverted Triangle", "Pear", "Hourglass", "Rectangle"]
BODY_SHAPES_MALE = ["Triangle", "Rectangle", "Trapezoid", "V-Taper"]
SKIN_UNDERTONES = ["Warm", "Cool", "Neutral"]

BADGE_CUSTOM = '<span class="badge custom"><i class="fa-solid fa-pen"></i> You customized</span>'


# ── Design system ───────────────────────────────────────────────────────────

def inject_theme_css() -> None:
    """Inject the MyType design system (fonts, tokens, components, responsive)."""
    st.markdown(
        """<style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

        :root {
          --ink:#16121F; --surface:#201A2E; --surface-2:#2A2240; --line:rgba(255,255,255,.08);
          --text:#F4EFFA; --muted:#A99FC2; --brand:#E91E8C; --brand-2:#F7941D;
          --ok:#3DD68C; --warn:#FFC24B; --err:#FF6B6B;
        }
        html, body, [class*="css"] { font-family:'Inter', sans-serif; }
        h1,h2,h3,h4,h5 { font-family:'Outfit', sans-serif; letter-spacing:-.01em; }
        .block-container { padding-top:2.2rem; padding-bottom:3rem; max-width:1180px; }

        /* Brand hero */
        .myt-hero { display:flex; align-items:center; gap:.9rem; margin-bottom:.2rem; }
        .myt-logo { width:46px; height:46px; border-radius:13px; display:grid; place-items:center;
          color:#fff; font-size:22px; background:linear-gradient(135deg, var(--brand), var(--brand-2));
          box-shadow:0 8px 24px rgba(233,30,140,.35); }
        .myt-hero h1 { margin:0; font-size:1.9rem; }
        .myt-tag { color:var(--muted); margin:.1rem 0 1.4rem; }

        /* Cards */
        div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px; border-color:var(--line) !important; }

        /* Trait cards */
        .trait { background:linear-gradient(160deg,var(--surface),var(--surface-2));
          border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; height:100%; }
        .trait .t-label { font-family:'Outfit'; font-weight:700; font-size:.78rem; letter-spacing:.07em;
          text-transform:uppercase; color:var(--muted); }
        .trait .t-value { font-family:'Outfit'; font-weight:800; font-size:1.2rem; margin:.2rem 0 .6rem; }

        .badge { display:inline-flex; align-items:center; gap:.35rem; font-size:.72rem; font-weight:600;
          padding:.22rem .6rem; border-radius:999px; margin-left:.35rem; }
        .badge.detected { background:rgba(61,214,140,.15); color:var(--ok); }
        .badge.custom   { background:rgba(255,194,75,.15); color:var(--warn); }
        .badge.low      { background:rgba(255,107,107,.15); color:var(--err); }

        .conf-track { height:6px; border-radius:99px; background:rgba(255,255,255,.08); overflow:hidden; }
        .conf-fill  { height:100%; border-radius:99px; background:linear-gradient(90deg,var(--brand),var(--brand-2)); }
        .conf-txt   { font-size:.76rem; color:var(--muted); display:flex; justify-content:space-between; margin-top:.35rem; }

        /* Upload dropzone + preview */
        [data-testid="stFileUploaderDropzone"] { border:1.5px dashed rgba(233,30,140,.5);
          border-radius:14px; background:rgba(233,30,140,.05); transition:border-color .2s ease, background .2s ease; }
        [data-testid="stFileUploaderDropzone"]:hover { border-color:var(--brand); background:rgba(233,30,140,.09); }
        .myt-preview { width:100%; max-height:200px; object-fit:cover; border-radius:12px;
          border:1px solid var(--line); margin-bottom:.4rem; }

        /* Recommendation cards */
        .rec-card { background:linear-gradient(165deg,var(--surface),var(--surface-2));
          border:1px solid var(--line); border-radius:16px; overflow:hidden; height:100%; }
        .rec-head { display:flex; align-items:center; gap:.6rem; padding:.85rem 1.1rem; color:#fff; }
        .rec-ic { width:32px; height:32px; border-radius:9px; background:rgba(255,255,255,.16);
          display:grid; place-items:center; font-size:15px; }
        .rec-title { font-family:'Outfit'; font-weight:700; font-size:1.05rem; }
        .rec-list { list-style:none; margin:0; padding:.9rem 1.1rem .4rem; }
        .rec-list li { position:relative; padding-left:1.15rem; margin-bottom:.5rem; color:var(--text); font-size:.92rem; }
        .rec-list li::before { content:'✦'; position:absolute; left:0; color:var(--brand); font-size:.75rem; top:.12rem; }
        .rec-src { margin:0 1.1rem 1rem; padding-top:.6rem; border-top:1px solid var(--line);
          font-size:.78rem; color:var(--muted); }

        /* Mobile: stack all columns */
        @media (max-width: 720px) {
          .block-container { padding-top:1.2rem; }
          .myt-hero h1 { font-size:1.4rem; }
          div[data-testid="stHorizontalBlock"] { flex-direction:column; gap:0; }
          div[data-testid="stHorizontalBlock"] > div { width:100% !important; min-width:100% !important; margin-bottom:1rem; }
          div[data-testid="stHorizontalBlock"] > div:last-child { margin-bottom:1; }
          div[data-testid="stHorizontalBlock"] > div:has(.trait),
          div[data-testid="stHorizontalBlock"] > div:has(.rec-card) { margin-bottom:1; }
          .trait, .rec-card { height:auto; margin-bottom:1rem; }
          .trait:last-child, .rec-card:last-child { margin-bottom:1; }
        }
        </style>""",
        unsafe_allow_html=True,
    )


# ── CV pipeline ─────────────────────────────────────────────────────────────

def _head_crop(image: np.ndarray, pose_landmarks) -> np.ndarray | None:
    """Crop and upscale the head region using pose landmarks (nose, ears, chest)."""
    height, width = image.shape[:2]
    lm = pose_landmarks.landmark
    xs = [lm[i].x for i in POSE_HEAD_LANDMARKS]
    ys = [lm[i].y for i in POSE_HEAD_LANDMARKS]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    box_w = max(max_x - min_x, max_y - min_y)
    if box_w <= 0:
        return None

    center_x = (min_x + max_x) / 2.0
    x0 = int(max(0.0, (center_x - box_w) * width))
    x1 = int(min(width, (center_x + box_w) * width))
    y0 = int(max(0.0, (min_y - box_w * 0.6) * height))
    y1 = int(min(height, (min_y + box_w * 1.6) * height))

    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    longest = max(crop.shape[:2])
    if longest < HEAD_CROP_SIZE:
        scale = HEAD_CROP_SIZE / float(longest)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


@st.cache_resource
def load_face_mesh():
    """Load the MediaPipe FaceMesh solution exactly once per session."""
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


@st.cache_resource
def load_pose():
    """Load the MediaPipe Pose solution exactly once per session."""
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    )


def process_image(image_bytes: bytes, gender: str = "female", progress=None):
    """Run the full CV pipeline without distorting body geometry through resize.

    Body pose analysis is performed on the original full-resolution image because
    aggressive resize can alter shoulder/hip proportions and misclassify body
    shapes in the web app. ``progress`` is an optional ``callable(stage: str)``
    used to animate a staged status widget during processing.
    """
    def _report(stage: str) -> None:
        if progress is not None:
            progress(stage)

    _report("Preparing your photo…")
    image = load_image(image_bytes)
    if image is None:
        return None

    body_image = image.copy()
    display_image = resize_to_max_dimension(image, MAX_IMAGE_DIMENSION)

    face_mesh = load_face_mesh()
    pose = load_pose()

    _report("Detecting body pose…")
    pose_results = pose.process(body_image)

    _report("Mapping facial landmarks…")
    face_result = face_analyzer.analyze_face(display_image, face_mesh)

    _report("Reading skin undertone…")
    skin_result = skin_analyzer.analyze_skin(display_image, face_mesh)

    if face_result.get("error") and pose_results.pose_landmarks:
        head = _head_crop(display_image, pose_results.pose_landmarks)
        if head is not None:
            face_result = face_analyzer.analyze_face(head, face_mesh)
            skin_result = skin_analyzer.analyze_skin(head, face_mesh)

    _report("Classifying body shape…")
    body_result = body_analyzer.analyze_body(body_image, pose, pose_results, gender=gender)

    return display_image, body_image, face_result, skin_result, body_result


# ── Presentation helpers ─────────────────────────────────────────────────────

def not_detected(result: dict) -> str:
    return "Not detected" if result.get("error") else result.get("shape", "Unknown")


def _build_options(detected: str, all_options: list[str]) -> tuple[list[str], int]:
    """Build a selectbox option list that always contains ``detected``."""
    options = list(all_options)
    if detected not in options:
        options = [detected] + options
    try:
        index = options.index(detected)
    except ValueError:
        index = 0
    return options, index


def skin_stability(skin_result: dict) -> str:
    """Heuristic stability label derived from the LAB channel standard deviations."""
    a_std = skin_result.get("a_std", 0) or 0
    b_std = skin_result.get("b_std", 0) or 0
    score = max(0, 100 - (a_std + b_std))
    if score >= 70:
        return "Stable"
    if score >= 45:
        return "Variable"
    return "Unstable"


def detected_badge(result: dict) -> str:
    """Confidence badge for face/body results (color + icon, never color-only)."""
    if result.get("error"):
        return '<span class="badge low"><i class="fa-solid fa-triangle-exclamation"></i> Not detected</span>'
    conf = result.get("confidence", 0)
    if conf >= 60:
        return f'<span class="badge detected"><i class="fa-solid fa-circle-check"></i> {conf:.0f}% confident</span>'
    return f'<span class="badge low"><i class="fa-solid fa-circle-minus"></i> {conf:.0f}% low confidence</span>'


def skin_badge(skin_result: dict) -> str:
    """Stability badge for the skin undertone (no probability is emitted today)."""
    if skin_result.get("error"):
        return '<span class="badge low"><i class="fa-solid fa-triangle-exclamation"></i> Not detected</span>'
    stability = skin_stability(skin_result)
    cls = {"Stable": "detected", "Variable": "custom", "Unstable": "low"}[stability]
    return f'<span class="badge {cls}"><i class="fa-solid fa-droplet"></i> {stability}</span>'


def trait_card(icon: str, label: str, value: str, badge_html: str, confidence: int | None) -> None:
    """Render a trait card with icon, value, badge and optional confidence bar."""
    conf = ""
    if confidence is not None:
        pct = min(100.0, max(0.0, confidence))
        conf = (f'<div class="conf-track"><div class="conf-fill" style="width:{pct:.0f}%"></div></div>'
                f'<div class="conf-txt"><span>Model confidence</span><span>{pct:.0f}%</span></div>')
    st.markdown(f"""
    <div class="trait">
      <div class="t-label">{ICON(icon)} {label} {badge_html}</div>
      <div class="t-value">{value}</div>
      {conf}
    </div>""", unsafe_allow_html=True)


def rec_card(icon: str, title: str, rule: str, source: str, accent: str) -> None:
    """Render a personalized recommendation card with visible source attribution."""
    items = "".join(f"<li>{t.strip()}</li>" for t in rule.split(","))
    st.markdown(f"""
    <div class="rec-card">
      <div class="rec-head" style="background:linear-gradient(135deg,{accent},rgba(0,0,0,.45))">
        <span class="rec-ic">{ICON(icon)}</span><span class="rec-title">{title}</span>
      </div>
      <ul class="rec-list">{items}</ul>
      <div class="rec-src"><i class="fa-solid fa-circle-info"></i> Recommended for your <b>{source}</b></div>
    </div>""", unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    """Apply the MyType chart styling to a plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, Inter, sans-serif", size=13, color="#A99FC2"),
        margin=dict(l=24, r=24, t=16, b=24),
    )
    return fig


def _go_to_recommendations() -> None:
    """Callback: switch the active tab to Your Recommendations."""
    st.session_state["active_tab"] = "Your Recommendations"


def _analyze_and_store(image_bytes: bytes, gender: str) -> None:
    """Run the staged CV pipeline with live status feedback and store the results."""
    status = st.status("Preparing your photo…", expanded=True)
    try:
        results = process_image(
            image_bytes,
            gender=gender,
            progress=lambda stage: status.update(label=stage, state="running"),
        )
        if results is None:
            status.update(label="Couldn't read that image — try a JPG or PNG.", state="error")
        else:
            st.session_state["results"] = results
            st.session_state["original_image"] = results[1]
            st.session_state["image_bytes"] = image_bytes
            st.session_state["analysis_gender"] = gender
            st.session_state["pending_tab"] = "Dashboard"
            status.update(label="Analysis complete — enjoy your results!", state="complete")
            st.rerun()
    except Exception:
        status.update(label="Analysis failed — try another photo.", state="error")
        raise


# ── App shell ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="MyType", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">',
    unsafe_allow_html=True,
)
inject_theme_css()

st.markdown(f"""
<div class="myt-hero">
  <div class="myt-logo">{ICON('wand-magic-sparkles')}</div>
  <h1>MyType</h1>
</div>
<div class="myt-tag">Upload a full-body or portrait photo and get an AI-powered style analysis.</div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(f"""
    <div class="myt-hero">
      <div class="myt-logo">{ICON('wand-magic-sparkles')}</div>
      <h1>MyType</h1>
    </div>
    <div class="myt-tag">Your personal AI style concierge</div>""", unsafe_allow_html=True)

    gender = st.segmented_control(
        "Gender",
        options=["Female", "Male"],
        default="Female",
        help="Tailors body-shape detection and grooming recommendations.",
    )

    st.divider()

    st.markdown(f"**{ICON('upload')} Upload a photo**", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a photo (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        mime = uploaded_file.type or "image/jpeg"
        b64 = base64.b64encode(uploaded_file.getvalue()).decode()
        st.markdown(
            f'<img src="data:{mime};base64,{b64}" class="myt-preview" alt="Uploaded photo preview" />',
            unsafe_allow_html=True,
        )

    process_clicked = st.button(
        "Analyze My Style",
        type="primary",
        width="stretch",
        icon=":material/auto_awesome:",
        disabled=uploaded_file is None,
        help="Runs the analysis locally on your device",
    )

    st.divider()

    if process_clicked and uploaded_file is not None:
        _analyze_and_store(uploaded_file.getvalue(), gender.lower())
    elif "results" in st.session_state:
        current_gender = gender.lower()
        if st.session_state.get("analysis_gender") != current_gender:
            stored = st.session_state["results"]
            status = st.status("Re-analyzing body shape…", expanded=True)
            try:
                display_image, body_image, face_result, skin_result, _ = stored
                pose = load_pose()
                status.update(label="Adjusting body-shape model…", state="running")
                pose_results = pose.process(body_image)
                new_body = body_analyzer.analyze_body(body_image, pose, pose_results, gender=current_gender)
                st.session_state["results"] = (display_image, body_image, face_result, skin_result, new_body)
                st.session_state["original_image"] = body_image
                st.session_state["analysis_gender"] = current_gender
                for key in ("custom_face", "custom_skin", "custom_body"):
                    st.session_state.pop(key, None)
                status.update(label="Body shape re-analyzed", state="complete")
                st.rerun()
            except Exception:
                status.update(label="Re-analysis failed", state="error")
                raise
        else:
            st.success("Analysis complete — view the tabs below.")
            if st.button("Clear Results", type="secondary", width="stretch"):
                for key in ("results", "original_image", "image_bytes",
                            "custom_face", "custom_skin", "custom_body"):
                    st.session_state.pop(key, None)
                st.rerun()

    with st.expander("📸 Photo tips", expanded=False):
        st.markdown(
            "- **Full body** in frame, standing upright\n"
            "- **Face the camera** directly\n"
            "- **Arms away** from your torso\n"
            "- **Simple background**, even lighting\n"
            "- **No filters or glasses** for skin tone"
        )
        st.markdown(
            "| ✅ Good | ❌ Avoid |\n"
            "|---|---|\n"
            "| Full body · upright · arms out | Cropped · seated · arms at sides |\n"
            "| Face-on · even light | Side angle · harsh shadows |\n"
            "| Plain background | Busy or mirror-selfie background |"
        )

results = st.session_state.get("results")

if "pending_tab" in st.session_state:
    st.session_state["active_tab"] = st.session_state.pop("pending_tab")

tab1, tab2, tab3 = st.tabs(
    ["Dashboard", "Advanced Analytics", "Your Recommendations"],
    key="active_tab",
    default="Dashboard",
    on_change="rerun",
)

# ── Dashboard ────────────────────────────────────────────────────────────────

with tab1:
    if results is None:
        st.markdown(f"### {ICON('upload')} Upload an Image", unsafe_allow_html=True)
        st.caption("Drag & drop or browse — JPG or PNG. Analysis starts automatically.")
        main_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            key="main_uploader",
            label_visibility="collapsed",
        )
        if main_file is not None:
            _analyze_and_store(main_file.getvalue(), gender.lower())

        st.markdown(f"### {ICON('wand-magic-sparkles')} How MyType works", unsafe_allow_html=True)
        st.markdown(
            f"1. {ICON('upload')} Upload a clear **portrait or full-body** photo in the sidebar.\n"
            f"2. {ICON('gear')} Click **Analyze My Style** — everything runs privately on your machine.\n"
            f"3. {ICON('chart-column')} Review your **Dashboard**, **Advanced Analytics** and **Recommendations**.\n\n"
            "> **Tip:** Good lighting and a straight-on pose give the most accurate results.",
            unsafe_allow_html=True,
        )
    else:
        image, original_image, face_result, skin_result, body_result = results

        face_detected = "Not detected" if face_result.get("error") else face_result.get("shape", "Unknown")
        skin_detected = "Not detected" if skin_result.get("error") else skin_result.get("undertone", "Unknown")
        body_detected = "Not detected" if body_result.get("error") else body_result.get("shape", "Unknown")

        face_options, face_idx = _build_options(face_detected, face_analyzer.FACE_SHAPES)
        skin_options, skin_idx = _build_options(skin_detected, SKIN_UNDERTONES)
        body_all = BODY_SHAPES_MALE if gender.lower() == "male" else BODY_SHAPES_FEMALE
        body_options, body_idx = _build_options(body_detected, body_all)

        if face_result.get("error"):
            st.warning(face_result["error"])
        if body_result.get("error"):
            st.warning(body_result["error"])

        st.markdown(f"<h3>{ICON('dna')} Style Profile</h3>", unsafe_allow_html=True)
        st.caption("Auto-detected from your photo — adjust any value and recommendations update instantly.")

        edit_cols = st.columns(3)
        with edit_cols[0]:
            face_choice = st.selectbox("Face Shape", face_options, index=face_idx, key="custom_face")
        with edit_cols[1]:
            skin_choice = st.selectbox("Skin Tone", skin_options, index=skin_idx, key="custom_skin")
        with edit_cols[2]:
            body_choice = st.selectbox("Body Shape", body_options, index=body_idx, key="custom_body")

        customized_face = face_choice != face_detected
        customized_skin = skin_choice != skin_detected
        customized_body = body_choice != body_detected

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        card_cols = st.columns(3)
        with card_cols[0]:
            trait_card(
                "user", "Face", face_choice,
                BADGE_CUSTOM if customized_face else detected_badge(face_result),
                face_result.get("confidence") if not face_result.get("error") else None,
            )
        with card_cols[1]:
            trait_card(
                "droplet", "Skin", skin_choice,
                BADGE_CUSTOM if customized_skin else skin_badge(skin_result),
                None,
            )
        with card_cols[2]:
            trait_card(
                "person", "Body", body_choice,
                BADGE_CUSTOM if customized_body else detected_badge(body_result),
                body_result.get("confidence") if not body_result.get("error") else None,
            )

        st.divider()

        if not face_result.get("error") and not body_result.get("error"):
            if face_result.get("confidence", 0) >= 60 and body_result.get("confidence", 0) >= 60:
                st.success("Everything looks great — your personalized recommendations are ready!")
            else:
                st.info("Some results have low confidence — recommendations may be generic.")

        st.button(
            "View Recommendations",
            type="primary",
            icon=":material/lightbulb:",
            width="stretch",
            on_click=_go_to_recommendations,
        )

# ── Advanced Analytics ───────────────────────────────────────────────────────

with tab2:
    if results is None:
        st.info("No analysis yet — process a photo to see the charts.")
    else:
        image, original_image, face_result, skin_result, body_result = results
        st.markdown(f"<h3>{ICON('chart-line')} Advanced Analytics</h3>", unsafe_allow_html=True)

        col_radar, col_body = st.columns(2)

        with col_radar:
            st.markdown(f"#### {ICON('bullseye')} Face Shape — Fuzzy Confidence", unsafe_allow_html=True)
            if face_result.get("error"):
                st.error("Face analysis failed — no radar chart available.")
            else:
                scores = face_result["scores"]
                order = sorted(scores, key=scores.get, reverse=True)
                radar = go.Figure(go.Scatterpolar(
                    r=[scores[s] for s in order], theta=order,
                    fill="toself", name="Confidence Score",
                    line_color="#e91e8c", fillcolor="rgba(233,30,140,0.30)",
                ))
                radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100],
                                               gridcolor="rgba(255,255,255,.12)",
                                               tickfont=dict(color="#A99FC2"))),
                    showlegend=False,
                )
                st.plotly_chart(style_fig(radar, 400), width="stretch")

        with col_body:
            st.markdown(f"#### {ICON('person')} Body Shape Confidence", unsafe_allow_html=True)
            if body_result.get("error"):
                st.error("Body analysis failed — no bar chart available.")
            else:
                shapes = list(body_result["scores"].keys())
                values = [round(v * 100, 1) for v in body_result["scores"].values()]
                colors = ["#f7941d" if s == body_result["shape"] else "#3a3350" for s in shapes]
                bar = go.Figure(go.Bar(
                    x=shapes, y=values,
                    marker_color=colors,
                    text=[f"{v:.0f}%" for v in values],
                    textposition="outside",
                ))
                bar.update_layout(
                    yaxis=dict(range=[0, 100], title="Confidence (%)", gridcolor="rgba(255,255,255,.12)"),
                    showlegend=False,
                )
                st.plotly_chart(style_fig(bar, 400), width="stretch")

        st.markdown("#### Deeper dive")

        col_top3, col_sil, col_skin = st.columns(3)

        with col_top3:
            st.markdown(f"##### {ICON('layer-group')} Top 3 Face Shapes", unsafe_allow_html=True)
            if face_result.get("error"):
                st.caption("Unavailable — face was not detected.")
            else:
                rows = sorted(face_result["top_shapes"], key=lambda r: r["confidence"], reverse=True)
                top3 = go.Figure(go.Bar(
                    y=[r["shape"] for r in rows],
                    x=[round(r["confidence"] * 100, 1) for r in rows],
                    orientation="h",
                    marker_color=["#e91e8c", "#f7941d", "#b0bec5"],
                    text=[f"{r['confidence']*100:.0f}%" for r in rows],
                    textposition="outside",
                ))
                top3.update_layout(
                    xaxis=dict(range=[0, 100], title="Normalized probability (%)", gridcolor="rgba(255,255,255,.12)"),
                    showlegend=False,
                )
                st.plotly_chart(style_fig(top3, 240), width="stretch")

        with col_sil:
            st.markdown(f"##### {ICON('ruler')} Body Widths (px)", unsafe_allow_html=True)
            if body_result.get("error"):
                st.caption("Unavailable — body was not detected.")
            else:
                hip = body_result.get("hip_width", 0)
                sh = body_result.get("shoulder_width", 0)
                labels, vals = ["Shoulders"], [sh]
                if body_result.get("w_to_h_ratio"):
                    labels.append("Waist")
                    vals.append(round(hip * body_result["w_to_h_ratio"], 1))
                labels.append("Hips")
                vals.append(hip)
                sil = go.Figure(go.Bar(
                    x=vals, y=labels, orientation="h",
                    marker_color="#e91e8c" if len(vals) == 2 else ["#e91e8c", "#f7941d", "#f7941d"],
                    text=[f"{v:.0f}px" for v in vals],
                    textposition="outside",
                ))
                sil.update_layout(xaxis=dict(gridcolor="rgba(255,255,255,.12)"), showlegend=False)
                st.plotly_chart(style_fig(sil, 240), width="stretch")

        with col_skin:
            st.markdown(f"##### {ICON('droplet')} Skin Undertone", unsafe_allow_html=True)
            if skin_result.get("error"):
                st.caption("Unavailable — skin region could not be extracted.")
            else:
                swatches = {"Warm": "#F7B267", "Cool": "#F2A9C7", "Neutral": "#E8B08C"}
                tone = skin_result.get("undertone", "Neutral")
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:1rem;background:linear-gradient(160deg,var(--surface),var(--surface-2));
                            border:1px solid var(--line);border-radius:16px;padding:1rem 1.2rem;height:150px;">
                  <div style="width:56px;height:56px;border-radius:50%;background:{swatches[tone]};
                              box-shadow:0 0 0 4px rgba(255,255,255,.08);"></div>
                  <div>
                    <div style="font-family:Outfit;font-weight:700;">{tone} undertone</div>
                    <div style="color:var(--muted);font-size:.82rem;margin-top:.25rem;">
                      Lab A* {skin_result.get('a_mean')} · B* {skin_result.get('b_mean')}<br>
                      Measurement: {skin_stability(skin_result).lower()}
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

        with st.expander("🔬 Raw extracted features (for the curious)"):
            st.caption("Measurements taken from your photo — lower values aren't 'better', they describe your geometry.")
            raw_data = pd.DataFrame(
                [
                    {
                        "Jaw Width (px)": face_result.get("features", {}).get("jaw_width"),
                        "Face Length (px)": face_result.get("features", {}).get("face_length"),
                        "Jaw Angle (deg)": face_result.get("features", {}).get("jaw_angle"),
                        "Width/Length Ratio": face_result.get("features", {}).get("w_to_l"),
                        "Forehead/Jaw Ratio": face_result.get("features", {}).get("forehead_to_jaw"),
                        "A* Mean (skin)": skin_result.get("a_mean"),
                        "B* Mean (skin)": skin_result.get("b_mean"),
                        "Shoulder/Hip Ratio": body_result.get("ratio"),
                    }
                ]
            )
            st.dataframe(raw_data, width="stretch", hide_index=True)

# ── Recommendations ──────────────────────────────────────────────────────────

with tab3:
    if results is None:
        st.info("No recommendations yet — process a photo first.")
    else:
        image, original_image, face_result, skin_result, body_result = results

        face_shape = st.session_state.get("custom_face") or not_detected(face_result)
        skin_tone = st.session_state.get("custom_skin") or skin_result.get("undertone", "Unknown")
        body_shape = st.session_state.get("custom_body") or not_detected(body_result)
        recs = recommender.get_recommendations(face_shape, skin_tone, body_shape, gender=gender.lower())

        st.markdown(f"<h3>{ICON('lightbulb')} Your Personalized Recommendations</h3>", unsafe_allow_html=True)
        st.caption(f"Tailored to your {face_shape} face, {skin_tone} undertone, and {body_shape} body.")

        is_male = gender.lower() == "male"
        style_key = "Skincare" if is_male else "Makeup"
        accent_map = {
            "Hairstyle": "#E91E8C",
            style_key: "#F7941D" if is_male else "#C8558C",
            "Outfit": "#7C5CFF",
        }
        icon_map = {
            "Hairstyle": "scissors",
            style_key: "spa" if is_male else "palette",
            "Outfit": "shirt",
        }

        col_h, col_m, col_o = st.columns(3)
        for col, key in zip([col_h, col_m, col_o], ["Hairstyle", style_key, "Outfit"]):
            with col:
                card = recs[key]
                rec_card(icon_map[key], key, card["rule"], card["source"], accent_map[key])

        st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

        if face_result.get("error") or body_result.get("error"):
            st.caption("Some traits weren't detected, so generic suggestions were used for those.")

        if (not face_result.get("error") and face_result.get("confidence", 0) < 60) or \
           (not body_result.get("error") and body_result.get("confidence", 0) < 60):
            st.warning("Some results have low confidence — suggestions may not be ideal for your body type.")