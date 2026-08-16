"""MyType - main Streamlit application entry point."""

from __future__ import annotations

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
import mediapipe as mp

from modules import face_analyzer, skin_analyzer, body_analyzer, recommender
from utils.helpers import load_image, resize_to_max_dimension

ICON = lambda name: f'<i class="fa-solid fa-{name}"></i>'

st.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">',
    unsafe_allow_html=True,
)

MAX_IMAGE_DIMENSION = 1024
HEAD_CROP_SIZE = 512
POSE_HEAD_LANDMARKS = [0, 7, 8, 2]


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


def process_image(image_bytes: bytes):
    """Run the full CV pipeline without distorting body geometry through resize.

    Body pose analysis is performed on the original full-resolution image because
    aggressive resize can alter shoulder/hip proportions and misclassify body
    shapes in the web app.
    """
    image = load_image(image_bytes)
    if image is None:
        return None

    body_image = image.copy()
    display_image = resize_to_max_dimension(image, MAX_IMAGE_DIMENSION)

    face_mesh = load_face_mesh()
    pose = load_pose()

    pose_results = pose.process(body_image)
    face_result = face_analyzer.analyze_face(display_image, face_mesh)
    skin_result = skin_analyzer.analyze_skin(display_image, face_mesh)

    if face_result.get("error") and pose_results.pose_landmarks:
        head = _head_crop(display_image, pose_results.pose_landmarks)
        if head is not None:
            face_result = face_analyzer.analyze_face(head, face_mesh)
            skin_result = skin_analyzer.analyze_skin(head, face_mesh)

    body_result = body_analyzer.analyze_body(body_image, pose, pose_results)

    return display_image, body_image, face_result, skin_result, body_result


def not_detected(result: dict) -> str:
    return "Not detected" if result.get("error") else result.get("shape", "Unknown")


def display_shape_with_confidence(result: dict, min_confidence: int = 60) -> str:
    """Format shape result with confidence indicator."""
    if result.get("error"):
        return "Not detected"
    shape = result.get("shape", "Unknown")
    confidence = result.get("confidence", 0)
    if confidence >= min_confidence:
        marker = "!!" if confidence >= 80 else "~"
        return f"{shape} ({confidence:.0f}%) {marker}"
    return f"{shape} ({confidence:.0f}%)"


st.set_page_config(page_title="MyType", layout="wide")
st.markdown(f"<h1>{ICON('wand-magic-sparkles')} MyType</h1>", unsafe_allow_html=True)
st.caption("Upload a full-body or portrait photo and get an AI-powered style analysis.")

with st.sidebar:
    st.markdown(f"<h3>{ICON('upload')} Upload</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
    process_clicked = st.button("Process Image", type="primary", width="stretch")
    st.divider()
    status_box = st.container()

    st.markdown("**Photo Tips**")
    st.markdown(
        "- Full-body, standing pose\n"
        "- Face the camera directly\n"
        "- Arms slightly away from torso\n"
        "- Simple background preferred\n"
        "- Good, even lighting"
    )

    if process_clicked:
        if uploaded_file is None:
            status_box.warning("Please upload a photo first.")
        else:
            image_bytes = uploaded_file.getvalue()
            with st.spinner("Analyzing your photo with MediaPipe..."):
                results = process_image(image_bytes)
            if results is None:
                status_box.error("Could not decode the image. Try another file.")
            else:
                st.session_state["results"] = results
                st.session_state["original_image"] = results[1]
                st.session_state["active_tab"] = 0
                status_box.success("Analysis complete!")
                st.rerun()
    elif "results" in st.session_state:
        status_box.success("Analysis complete! View the tabs below.")
        if st.button("Clear Results", type="secondary"):
            del st.session_state["results"]
            if "original_image" in st.session_state:
                del st.session_state["original_image"]
            st.rerun()

results = st.session_state.get("results")

active_tab = st.session_state.get("active_tab", 0)
tab1, tab2, tab3 = st.tabs(["Dashboard", "Advanced Analytics", "Your Recommendations"])

with tab1:
    if results is None:
        st.info("Upload a photo and click **Process Image** to start the analysis.")
        st.markdown(
            "### How it works\n\n"
            f"1. {ICON('upload')} Upload a clear **portrait or full-body** photo in the sidebar.\n"
            f"2. {ICON('gear')} Click **Process Image** - MediaPipe runs fully on your machine.\n"
            f"3. {ICON('chart-column')} Review your **Dashboard**, **Advanced Analytics** and **Recommendations**.\n\n"
            "> **Tip:** Use good lighting and a straight-on pose for the most accurate results.",
            unsafe_allow_html=True,
        )
    else:
        image, original_image, face_result, skin_result, body_result = results
        col_image, col_metrics = st.columns([1, 1], gap="large")

        with col_image:
            st.markdown(f"<h3>{ICON('image')} Your Photo</h3>", unsafe_allow_html=True)
            
            view_mode = st.radio(
                "View",
                ["Processed", "Original"],
                horizontal=True,
                label_visibility="collapsed",
            )
            
            if view_mode == "Original" and "original_image" in st.session_state:
                st.image(st.session_state["original_image"], caption="Original image", width="stretch")
            else:
                st.image(image, caption="Processed image (auto-resized)", width="stretch")
            
            if face_result.get("error"):
                st.warning(face_result["error"])
            if body_result.get("error"):
                st.warning(body_result["error"])

        with col_metrics:
            st.markdown(f"<h3>{ICON('dna')} Style Profile</h3>", unsafe_allow_html=True)
            st.metric("Face Shape", display_shape_with_confidence(face_result))
            st.metric("Skin Tone", skin_result.get("undertone", "Not detected"))
            st.metric("Body Shape", display_shape_with_confidence(body_result))

            if not face_result.get("error") and not body_result.get("error"):
                face_conf = face_result.get("confidence", 0)
                body_conf = body_result.get("confidence", 0)
                if face_conf >= 60 and body_conf >= 60:
                    st.markdown(f'<div class="stAlert stSuccess"><p>Everything looks great! Head over to the Recommendations tab. {ICON("arrow-down")}</p></div>', unsafe_allow_html=True)
                else:
                    st.info("Some results have low confidence. Recommendations may be generic.")

with tab2:
    if results is None:
        st.info("No analysis yet. Process an image to see the charts.")
    else:
        image, original_image, face_result, skin_result, body_result = results

        st.markdown(f"<h3>{ICON('chart-line')} Advanced Analytics</h3>", unsafe_allow_html=True)

        radar_col, bar_col = st.columns(2)
        with radar_col:
            st.markdown(f"#### {ICON('bullseye')} Face Shape Fuzzy Confidence", unsafe_allow_html=True)
            if face_result.get("error"):
                st.error("Face analysis failed - no radar chart available.")
            else:
                shapes = face_analyzer.FACE_SHAPES
                scores = [face_result["scores"][s] for s in shapes]
                radar = go.Figure(
                    go.Scatterpolar(
                        r=scores,
                        theta=shapes,
                        fill="toself",
                        name="Confidence Score",
                        line_color="#e91e8c",
                        fillcolor="rgba(233,30,140,0.30)",
                    )
                )
                radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    height=400,
                    margin=dict(l=40, r=40, t=30, b=30),
                )
                st.plotly_chart(radar, width="stretch")

        with bar_col:
            st.markdown(f"#### {ICON('person')} Body Shape Confidence", unsafe_allow_html=True)
            if body_result.get("error"):
                st.error("Body analysis failed - no bar chart available.")
            else:
                shapes = list(body_result["scores"].keys())
                values = [round(v * 100, 1) for v in body_result["scores"].values()]
                colors = [
                    "#f7941d" if s == body_result["shape"] else "#b0bec5"
                    for s in shapes
                ]
                bar = go.Figure(
                    go.Bar(
                        x=shapes,
                        y=values,
                        marker_color=colors,
                        text=[f"{v:.0f}%" for v in values],
                        textposition="outside",
                    )
                )
                bar.update_layout(
                    yaxis=dict(range=[0, 100], title="Confidence (%)"),
                    showlegend=False,
                    height=400,
                    margin=dict(l=40, r=40, t=30, b=30),
                )
                st.plotly_chart(bar, width="stretch")

        st.markdown(f"#### {ICON('microscope')} Extracted Raw Features", unsafe_allow_html=True)
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

with tab3:
    if results is None:
        st.info("No recommendations yet. Process an image first.")
    else:
        image, original_image, face_result, skin_result, body_result = results

        face_shape = not_detected(face_result)
        skin_tone = skin_result.get("undertone", "Unknown")
        body_shape = not_detected(body_result)
        recs = recommender.get_recommendations(face_shape, skin_tone, body_shape)

        st.markdown(f"<h3>{ICON('lightbulb')} Your Personalized Recommendations</h3>", unsafe_allow_html=True)

        icon_map = {"Hairstyle": ICON("scissors"), "Makeup": ICON("palette"), "Outfit": ICON("shirt")}
        col_h, col_m, col_o = st.columns(3)

        for col, key in zip([col_h, col_m, col_o], ["Hairstyle", "Makeup", "Outfit"]):
            with col:
                card = recs[key]
                with st.container(border=True):
                    st.markdown(f"### {icon_map[key]} {key}")
                    st.markdown(f"*Based on your: **{card['source']}***")
                    for item in card["rule"].split(","):
                        st.markdown(f"- {item.strip()}")

        if face_result.get("error") or body_result.get("error"):
            st.caption("Some traits were not detected, so generic suggestions were used for those.")

        if (not face_result.get("error") and face_result.get("confidence", 0) < 60) or \
           (not body_result.get("error") and body_result.get("confidence", 0) < 60):
            st.warning("Some results have low confidence. Suggestions may not be ideal for your body type.")