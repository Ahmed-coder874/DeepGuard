"""
DeepGuard - AI-Based Deepfake Detection Prototype
=================================================

This is the Streamlit user interface for the first prototype of the project
"Deepfake Technology" (Project ID 24CGCS06).

What this version does:
    - Accepts an image upload (JPG / JPEG / PNG)
    - Validates the uploaded file
    - Preprocesses the image (resize to 224 x 224, RGB conversion, normalisation)
    - Displays the original and the preprocessed image
    - Runs the trained EfficientNet-B0 model on the image and shows a
      REAL / FAKE prediction together with the model confidence

The prediction is produced by the real trained checkpoint stored at
models/deepguard_efficientnet_b0.pt and is run through the existing
src/inference module. It is an AI model output, so it must be treated as an
indication requiring further verification, not as proof of authenticity.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys

import streamlit as st

# Make sure the local "src" package can be imported no matter which folder the
# command is run from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from src import dataset, inference, model_interface, preprocessing, validation  # noqa: E402


# ---------------------------------------------------------------------------
# Page configuration and styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepGuard - Deepfake Detection Prototype",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

    .dg-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        padding: 1.75rem 2rem;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 1.5rem;
    }
    .dg-header h1 { margin: 0; font-size: 2.2rem; color: #ffffff; }
    .dg-header p { margin: 0.35rem 0 0 0; font-size: 1.05rem; opacity: 0.96; }

    .dg-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.9rem 1.05rem;
        height: 100%;
    }
    .dg-card .dg-label {
        font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em;
        color: #64748b;
    }
    .dg-card .dg-value {
        font-size: 1.02rem; font-weight: 600; color: #0f172a; margin-top: 0.2rem;
    }

    .dg-pipeline {
        display: flex; flex-direction: column; align-items: center; gap: 0.15rem;
    }
    .dg-step {
        width: 100%; max-width: 470px; text-align: center;
        padding: 0.55rem 1rem; border-radius: 10px;
        font-weight: 600; font-size: 0.95rem; border: 1px solid transparent;
    }
    .dg-step-done { background: #dcfce7; color: #166534; border-color: #86efac; }
    .dg-step-pending { background: #f1f5f9; color: #64748b; border-color: #cbd5e1; }
    .dg-arrow { color: #94a3b8; font-size: 1.05rem; line-height: 1; }

    .dg-notice {
        background: #fffbeb; border: 1px solid #fde68a; border-left: 5px solid #f59e0b;
        border-radius: 10px; padding: 0.9rem 1.1rem; color: #78350f;
    }

    .dg-footer { text-align: center; color: #94a3b8; font-size: 0.8rem; margin-top: 2rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small display helpers
# ---------------------------------------------------------------------------
def info_card(label: str, value: str) -> str:
    """Return the HTML for one small information card."""
    return (
        f'<div class="dg-card"><div class="dg-label">{label}</div>'
        f'<div class="dg-value">{value}</div></div>'
    )


def render_pipeline() -> None:
    """Draw the proposed pipeline and show which stages are implemented."""
    steps = [
        ("Input", True),
        ("Validation", True),
        ("Preprocessing", True),
        ("Feature / Representation Preparation", True),
        ("AI / ML Model", True),
        ("Classification", True),
        ("Prediction", True),
    ]

    parts = ['<div class="dg-pipeline">']
    for index, (label, done) in enumerate(steps):
        css_class = "dg-step-done" if done else "dg-step-pending"
        mark = "&#10003;" if done else "&#8987;"
        suffix = "" if done else " - pending"
        parts.append(
            f'<div class="dg-step {css_class}">{mark} {label}{suffix}</div>'
        )
        if index < len(steps) - 1:
            parts.append('<div class="dg-arrow">&#8595;</div>')
    parts.append("</div>")

    st.markdown("".join(parts), unsafe_allow_html=True)
    st.caption(
        "All seven steps are implemented in this prototype. The real AI/ML "
        "prediction comes from the trained EfficientNet-B0 checkpoint."
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## DeepGuard")
    st.caption("AI-Based Deepfake Detection Prototype")
    st.divider()

    st.markdown("### Project Information")
    st.markdown(
        "- **Project:** Deepfake Technology\n"
        "- **Project ID:** 24CGCS06\n"
        "- **SDG:** 10\n"
        "- **Department:** ECE\n"
        "- **Project Guide:** Prof. Jayanth U"
    )
    st.divider()

    st.markdown("### Current Prototype Stage")
    st.markdown(
        "- ✓ Image upload\n"
        "- ✓ Input validation\n"
        "- ✓ Image preprocessing\n"
        "- ✓ Processed-image visualization\n"
        "- ✓ Dataset research and preparation tools\n"
        "- ✓ Model architecture selection (EfficientNet-B0)\n"
        "- ✓ Trained deepfake detection model (EfficientNet-B0)\n"
        "- ✓ Dataset-based prediction with confidence"
    )

    if model_interface.is_model_available():
        st.success("The trained model is integrated and predictions are live.")
    else:
        st.error(
            "The checkpoint file is not present in this clone. Place "
            "`models/deepguard_efficientnet_b0.pt` in the project folder to "
            "enable predictions."
        )
    st.divider()

    st.markdown("### Future Development")
    st.write(
        "The trained model is integrated and produces REAL/FAKE predictions "
        "for uploaded images. Future work covers video and audio analysis, "
        "deployment and further evaluation."
    )
    st.caption("Prototype build - trained model integrated.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="dg-header">
        <h1>DeepGuard</h1>
        <p>AI-Based Deepfake Detection Prototype</p>
        <p style="font-size:0.95rem;">
            An initial prototype for analyzing digital media as part of an
            AI/ML-based deepfake detection system.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(info_card("Project", "Deepfake Technology"), unsafe_allow_html=True)
with col2:
    st.markdown(info_card("Project ID", "24CGCS06"), unsafe_allow_html=True)
with col3:
    st.markdown(
        info_card("SDG", "10 - Cybersecurity &amp; Digital Governance"),
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        info_card(
            "Current Prototype Stage",
            "Trained model integrated - REAL/FAKE predictions",
        ),
        unsafe_allow_html=True,
    )

st.divider()


# ---------------------------------------------------------------------------
# Section 1 - Upload
# ---------------------------------------------------------------------------
st.subheader("1. Upload an Image")
st.info(
    "Current prototype supports image input. Video and audio analysis are "
    "planned for future development."
)

st.caption(
    "Demo guidance: use your own separate demonstration images. The official "
    "120-image held-out test set under `data/processed/test` is part of the "
    "frozen evaluation and must not be used as casual demo material."
)

uploaded_file = st.file_uploader(
    "Choose an image file",
    accept_multiple_files=False,
    help=(
        "Accepted formats: JPG, JPEG and PNG. Any other file type is rejected "
        "by the validation step with a friendly message."
    ),
)


# ---------------------------------------------------------------------------
# Section 2 - Validation, original image and preprocessing
# ---------------------------------------------------------------------------
image_info = None
image_bytes = b""
current_key = ""

if uploaded_file is None:
    st.warning("No file selected yet. Please upload a JPG or PNG image to continue.")
else:
    image_bytes = uploaded_file.getvalue()
    current_key = f"{uploaded_file.name}|{uploaded_file.size}"

    # If the user uploads a different file, forget the previous analysis.
    if st.session_state.get("analysis_key") != current_key:
        st.session_state["analysis"] = None
        st.session_state["detection"] = None
        st.session_state["analysis_key"] = current_key

    # --- Input validation -------------------------------------------------
    try:
        image_info = validation.validate_uploaded_file(
            uploaded_file.name,
            uploaded_file.type,
            image_bytes,
        )
    except validation.ImageValidationError as error:
        st.error(f"⚠ Unable to process this image. {error}")
    except Exception:
        st.error(
            "⚠ Unable to process this image because of an unexpected problem. "
            "Please try a different JPG or PNG file."
        )


if image_info is not None:
    st.success("✓ Image successfully loaded")

    # --- Original image display ------------------------------------------
    st.markdown("#### Original Image")
    original_col, details_col = st.columns([1.3, 1])

    with original_col:
        st.image(image_bytes, caption=image_info.filename, width="stretch")

    with details_col:
        st.markdown("**File details**")
        st.write(f"**Filename:** {image_info.filename}")
        st.write(
            f"**File type:** {image_info.image_format} ({image_info.mime_type})"
        )
        st.write(f"**File size:** {image_info.size_kb} KB")
        st.write(f"**Original width:** {image_info.width} px")
        st.write(f"**Original height:** {image_info.height} px")
        st.write(f"**Channels:** {image_info.channels} (mode: {image_info.mode})")

    st.divider()

    # --- Preprocessing + prediction trigger ------------------------------
    st.subheader("2. Analyze Image")
    st.caption(
        "Clicking the button validates the file, runs the real preprocessing "
        "and then runs the trained EfficientNet-B0 model. The result is a "
        "REAL / FAKE prediction with the model's confidence."
    )

    if st.button("Analyze Image", type="primary", width="stretch"):
        try:
            with st.spinner("Running the trained model..."):
                result = preprocessing.preprocess_image(image_bytes)
                prediction = inference.predict_image(result["rgb"])
            st.session_state["analysis"] = result
            st.session_state["detection"] = prediction
            st.session_state["analysis_key"] = current_key
        except preprocessing.PreprocessingError as error:
            st.session_state["analysis"] = None
            st.session_state["detection"] = None
            st.error(f"⚠ Preprocessing failed. {error}")
        except inference.InferenceNotAvailableError as error:
            st.session_state["detection"] = None
            st.error(f"⚠ Detection model unavailable. {error}")
        except Exception:
            st.session_state["analysis"] = None
            st.session_state["detection"] = None
            st.error("⚠ Unable to analyze this image because of an unexpected problem.")

    # --- Preprocessed image display --------------------------------------
    result = st.session_state.get("analysis")
    if result is not None and st.session_state.get("analysis_key") == current_key:
        st.success("✓ Preprocessing completed successfully")

        st.markdown("#### Preprocessed Image")
        processed_col, processed_details = st.columns([1.3, 1])

        with processed_col:
            st.image(
                result["resized_rgb"],
                caption="Preprocessed image (224 x 224)",
                width="stretch",
            )

        with processed_details:
            original_width, original_height = result["original_size"]
            processed_width, processed_height = result["processed_size"]
            st.markdown("**Preprocessing details**")
            st.write(
                f"**Original dimensions:** {original_width} x {original_height} px"
            )
            st.write(
                f"**Processed dimensions:** {processed_width} x {processed_height} px"
            )
            st.write("**Pixel normalization:** 0 - 1")
            st.write(f"**Data type:** {result['normalized'].dtype}")
            st.write(f"**Model input shape:** {result['model_input_shape']}")

        with st.expander("What did the preprocessing do?"):
            st.markdown(
                "- Decoded the uploaded file into an image\n"
                "- Converted the colour order from BGR (OpenCV) to RGB\n"
                "- Resized the image to 224 x 224 pixels\n"
                "- Normalised pixel values from 0-255 to the range 0-1\n"
                "- Added a batch dimension, giving an array of shape "
                "(1, 224, 224, 3)\n\n"
                "This step only **prepares** the image for display. The model "
                "input used by the trained network is the normalised "
                "(1, 3, 224, 224) tensor built inside `src/inference`."
            )

        # --- Detection result (real model output) -----------------------
        st.markdown("#### Detection Result")
        detection = st.session_state.get("detection")
        if detection is None:
            st.warning(
                "No prediction is shown because the detection model could not "
                "be run for this image."
            )
        else:
            label = detection["prediction_class"].upper()
            confidence = detection["confidence"]
            p_real = detection["probabilities"]["real"]
            p_fake = detection["probabilities"]["fake"]

            if label == "FAKE":
                st.error(f"**Prediction: {label}**")
            else:
                st.success(f"**Prediction: {label}**")
            st.write(
                f"**Model confidence:** {confidence:.2%} "
                "(probability of the predicted class)"
            )
            st.progress(float(confidence))
            st.write(
                f"**Model output:** real = {p_real:.2%}, fake = {p_fake:.2%}"
            )
            st.write(f"**Model:** EfficientNet-B0 (PyTorch)")
            st.write(f"**Checkpoint:** {detection['checkpoint_path']}")
            st.caption(
                "This is an AI model prediction for one uploaded image, not "
                "proof of authenticity. It should be treated as an indication "
                "requiring further verification. It is a live demo output and "
                "is separate from the frozen official evaluation of the "
                "120-image held-out test set."
            )

    st.divider()


# ---------------------------------------------------------------------------
# Section 3 - Pipeline
# ---------------------------------------------------------------------------
st.subheader("3. Processing Pipeline")
st.write(
    "The deepfake detection pipeline. All stages are implemented; the "
    "AI/ML prediction is produced by the trained EfficientNet-B0 model."
)
render_pipeline()

st.divider()


# ---------------------------------------------------------------------------
# Section 4 - Model status
# ---------------------------------------------------------------------------
st.subheader("4. Detection Model Status")

model_status = model_interface.get_model_status()

if model_status["available"]:
    st.success("AI/ML detection model is integrated and active.")
else:
    st.error(
        "No trained checkpoint was found, so the model cannot be loaded and no "
        "prediction can be produced."
    )
    st.code(
        "Place the model file at:\n"
        f"{config.PROJECT_ROOT}\\models\\{config.CHECKPOINT_FILENAME}",
        language="text",
    )

model_col1, model_col2, model_col3 = st.columns(3)
with model_col1:
    st.markdown(
        info_card("Selected Architecture", model_status["architecture"]),
        unsafe_allow_html=True,
    )
with model_col2:
    st.markdown(
        info_card("Framework", "PyTorch (CPU)"), unsafe_allow_html=True
    )
with model_col3:
    st.markdown(
        info_card(
            "Trained Model",
            "Available (integrated)" if model_status["available"] else "Not found",
        ),
        unsafe_allow_html=True,
    )

st.write(model_status["message"])
st.caption(f"Checkpoint location: {model_status['checkpoint_path']}")

report_path = config.get_evaluation_report_path()
if report_path.is_file():
    evaluation_report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = evaluation_report.get("metrics", {})
    accuracy = metrics.get("accuracy")
    num_samples = metrics.get("num_samples")
    macro_f1 = metrics.get("macro", {}).get("f1")
    if accuracy is not None and num_samples is not None:
        st.markdown(
            f"**Official evaluation (recorded Stage 5 result):** "
            f"{accuracy:.2%} accuracy on {num_samples} test images "
            f"(macro F1 {macro_f1:.4f}). See `reports/evaluation_report.json`. "
            "This recorded test result is based on the 120-image test split "
            "and is separate from the live single-image prediction shown in "
            "section 2."
        )
else:
    st.write(
        "No recorded evaluation report was found, so no official metric is "
        "reported by this application."
    )

st.divider()


# ---------------------------------------------------------------------------
# Section 5 - Dataset status
# ---------------------------------------------------------------------------
st.subheader("5. Dataset Status")
st.write(
    "This section reports the real contents of the dataset folders on disk. "
    "Nothing here is estimated or invented."
)

dataset_status = dataset.get_dataset_status()
summary = dataset_status["summary"]

if summary is not None:
    totals = summary.get("totals", {})
    split_info = summary.get("splits", {})

    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        st.metric("Real images", totals.get("real", 0))
    with status_col2:
        st.metric("Fake images", totals.get("fake", 0))
    with status_col3:
        st.metric("Invalid / corrupt files", totals.get("invalid", 0))

    status_col4, status_col5, status_col6 = st.columns(3)
    with status_col4:
        st.metric("Training images", totals.get("train", 0))
    with status_col5:
        st.metric("Validation images", totals.get("validation", 0))
    with status_col6:
        st.metric("Test images", totals.get("test", 0))

    st.success("✓ Prepared dataset found.")
    st.caption(
        f"Prepared at {summary.get('generated_at', 'unknown')} | "
        f"seed={summary.get('seed', '?')} | "
        f"max per class={summary.get('max_per_class')} | "
        f"unsupported files={totals.get('unsupported', 0)}"
    )

    with st.expander("Split details"):
        for split_name in ("train", "validation", "test"):
            info = split_info.get(split_name, {})
            st.write(
                f"**{split_name.capitalize()}:** "
                f"{info.get('real', 0)} real + {info.get('fake', 0)} fake "
                f"= {info.get('total', 0)} images"
            )

elif dataset_status["raw_real_count"] > 0 or dataset_status["raw_fake_count"] > 0:
    st.warning(
        "Raw images were found, but the dataset has not been prepared yet."
    )
    st.write(f"**Raw real images:** {dataset_status['raw_real_count']}")
    st.write(f"**Raw fake images:** {dataset_status['raw_fake_count']}")
    st.info(
        "Run  `python prepare_dataset.py`  in the project folder to create the "
        "train / validation / test splits."
    )

else:
    st.warning("No dataset found yet.")
    st.write("DeepGuard expects the images to be placed in these folders:")
    st.code(
        "data/raw/real/   <- genuine images\n"
        "data/raw/fake/   <- manipulated / AI-generated images",
        language="text",
    )
    st.info(
        "Download a public deepfake dataset (see docs/dataset_research.md), "
        "place the images in the folders above, then run "
        "`python prepare_dataset.py`."
    )

st.caption(
    "Dataset preparation organises and checks images only. It does not train "
    "a model and it does not detect deepfakes."
)

st.divider()


# ---------------------------------------------------------------------------
# Section 6 - Planned next steps
# ---------------------------------------------------------------------------
st.subheader("6. Planned Next Steps")
st.markdown(
    "1. ✓ Study publicly available deepfake datasets - documented in "
    "`docs/dataset_research.md`.\n"
    "2. ✓ Finalize the supported media type and detection approach "
    "(single-image classification).\n"
    "3. ✓ Study suitable deepfake detection architectures and select one - "
    "EfficientNet-B0, documented in `docs/model_architecture_research.md`.\n"
    "4. ✓ Prepare the selected dataset (300 real + 300 fake frames from a "
    "20-video FaceForensics++ experiment, leak-free splits).\n"
    "5. ✓ Implement and run model training (`src/training.py`, "
    "`train_model.py`).\n"
    "6. ✓ Validate the trained model.\n"
    "7. ✓ Evaluate using accuracy, precision, recall and F1-score "
    "(see `reports/evaluation_report.json`).\n"
    "8. ✓ Integrate the trained model with DeepGuard.\n"
    "9. ✓ Perform unit and integration testing.\n"
    "10. Extend the system to additional media types if feasible "
    "(planned future work)."
)
st.caption(
    "Items 1-9 are complete. Item 10 (video and audio analysis) is planned "
    "future work beyond this prototype."
)

st.divider()


# ---------------------------------------------------------------------------
# Section 7 - Societal context
# ---------------------------------------------------------------------------
st.subheader("7. Societal Context")
st.write(
    "A field visit carried out for this project identified concerns related "
    "to deepfake-related cybercrime, identity fraud, misinformation, the "
    "difficulty of verifying digital content, and limited public awareness of "
    "deepfake risks."
)
st.write(
    "This prototype is intended as an initial technical step towards helping "
    "users analyze potentially manipulated digital content. It focuses on "
    "building a reliable input and preprocessing stage on which a detection "
    "model can later be based."
)
st.caption(
    "No statistics are shown here because no verified source has been "
    "provided for them."
)

st.divider()


# ---------------------------------------------------------------------------
# Responsible AI notice
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="dg-notice">
        <strong>Important:</strong> Deepfake detection systems are not
        guaranteed to be perfectly accurate. Any future prediction should be
        treated as an indication requiring further verification rather than
        absolute proof of authenticity.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dg-footer">DeepGuard - Deepfake Technology (24CGCS06) - '
    "Atria Institute of Technology - Prototype: trained model integrated</div>",
    unsafe_allow_html=True,
)
