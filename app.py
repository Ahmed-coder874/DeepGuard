"""
DeepGuard - AI-Based Deepfake Detection - Experiment 2 Prototype
================================================================

This is the Streamlit user interface for the project "Deepfake Technology"
(Project ID 24CGCS06), presenting the completed Experiment 2.

What this version does:
    - Accepts an image upload (JPG / JPEG / PNG)
    - Validates the uploaded file
    - Preprocesses the image (resize to 224 x 224, RGB conversion, normalisation)
    - Displays the original and the preprocessed image
    - Runs the trained EfficientNet-B0 model on the image and shows a
      REAL / FAKE prediction together with the model confidence
    - Reports the verified Experiment 2 dataset, training and final test
      results, read from the committed Experiment 2 result files

The prediction is produced by the real Experiment 2 trained checkpoint stored
at models/experiment_2_deepguard_efficientnet_b0.pt and is run through the
existing src/inference module. It is an AI model output, so it must be treated
as an indication requiring further verification, not as proof of authenticity.

Experiment 1 is retained unchanged as a historical record and is NOT the model
this prototype loads.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

# Make sure the local "src" package can be imported no matter which folder the
# command is run from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from src import dataset, inference, model_interface, preprocessing, validation  # noqa: E402


# ---------------------------------------------------------------------------
# Experiment 2 configuration
# ---------------------------------------------------------------------------
# Experiment 2 is the active experiment of this prototype. It is selected here,
# explicitly and locally, so that config.py (which still describes the frozen
# Experiment 1 defaults) does not have to be modified. The Experiment 1
# checkpoint stays in models/ untouched.
EXPERIMENT_LABEL = "Experiment 2"
EXPERIMENT_CHECKPOINT_FILENAME = "experiment_2_deepguard_efficientnet_b0.pt"

EXPERIMENT_DIR = config.PROJECT_ROOT / "experiments" / "experiment_2_large_dataset"
EXPERIMENT_CHECKPOINT_PATH = config.CHECKPOINT_DIR / EXPERIMENT_CHECKPOINT_FILENAME
EXPERIMENT_PROCESSED_DIR = EXPERIMENT_DIR / "processed"
EXPERIMENT_DATASET_REPORT_PATH = EXPERIMENT_DIR / "dataset_report.json"
EXPERIMENT_TRAINING_SUMMARY_PATH = EXPERIMENT_DIR / "training_summary.json"
EXPERIMENT_EVALUATION_REPORT_PATH = (
    EXPERIMENT_DIR / "reports" / "evaluation_report.json"
)


def read_json_file(path: Path):
    """Return the parsed JSON at ``path``, or None when it is absent/unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Page configuration and styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepGuard - Deepfake Detection Experiment 2 Prototype",
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
    st.caption("AI-Based Deepfake Detection - Experiment 2 Prototype")
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
        "- ✓ Experiment 2 dataset preparation and near-duplicate auditing\n"
        "- ✓ Model architecture selection (EfficientNet-B0)\n"
        "- ✓ Experiment 2 trained deepfake detection model (EfficientNet-B0)\n"
        "- ✓ Dataset-based prediction with confidence"
    )

    if model_interface.is_model_available(filename=EXPERIMENT_CHECKPOINT_FILENAME):
        st.success(
            f"The {EXPERIMENT_LABEL} model is integrated and predictions are live."
        )
    else:
        st.error(
            "The Experiment 2 checkpoint file is not present in this clone. "
            f"Place `models/{EXPERIMENT_CHECKPOINT_FILENAME}` in the project "
            "folder to enable predictions."
        )
    st.divider()

    st.markdown("### Future Development")
    st.write(
        "The Experiment 2 model is integrated and produces REAL/FAKE "
        "predictions for uploaded images. Future work covers video and audio "
        "analysis, deployment and further evaluation."
    )
    st.caption(f"Prototype build - {EXPERIMENT_LABEL} model integrated.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="dg-header">
        <h1>DeepGuard</h1>
        <p>AI-Based Deepfake Detection - Experiment 2 Prototype</p>
        <p style="font-size:0.95rem;">
            An AI/ML-based deepfake detection system running the completed
            Experiment 2 EfficientNet-B0 model.
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
            f"{EXPERIMENT_LABEL} model integrated - REAL/FAKE predictions",
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
    "900-image held-out test set under "
    "`experiments/experiment_2_large_dataset/processed/test` is part of the "
    "frozen Experiment 2 evaluation and must not be used as casual demo "
    "material."
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
        "and then runs the Experiment 2 trained EfficientNet-B0 model. The "
        "result is a REAL / FAKE prediction with the model's confidence."
    )

    if st.button("Analyze Image", type="primary", width="stretch"):
        try:
            with st.spinner("Running the trained model..."):
                result = preprocessing.preprocess_image(image_bytes)
                prediction = inference.predict_image(
                    result["rgb"], filename=EXPERIMENT_CHECKPOINT_FILENAME
                )
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
            st.write(f"**Experiment:** {EXPERIMENT_LABEL}")
            st.write(f"**Checkpoint:** {EXPERIMENT_CHECKPOINT_FILENAME}")
            st.caption(
                "This is an AI model prediction for one uploaded image, not "
                "proof of authenticity. It should be treated as an indication "
                "requiring further verification. It is a live demo output and "
                "is separate from the frozen Experiment 2 evaluation of the "
                "900-image held-out test set."
            )

    st.divider()


# ---------------------------------------------------------------------------
# Section 3 - Pipeline
# ---------------------------------------------------------------------------
st.subheader("3. Processing Pipeline")
st.write(
    "The deepfake detection pipeline. All stages are implemented; the "
    "AI/ML prediction is produced by the Experiment 2 trained EfficientNet-B0 "
    "model."
)
render_pipeline()

st.divider()


# ---------------------------------------------------------------------------
# Section 4 - Model status
# ---------------------------------------------------------------------------
st.subheader("4. Detection Model Status")

model_status = model_interface.get_model_status(
    filename=EXPERIMENT_CHECKPOINT_FILENAME
)

if model_status["available"]:
    st.success(
        f"AI/ML detection model ({EXPERIMENT_LABEL}) is integrated and active."
    )
else:
    st.error(
        f"No trained {EXPERIMENT_LABEL} checkpoint was found, so the model "
        "cannot be loaded and no prediction can be produced."
    )
    st.code(
        "Place the model file at:\n"
        f"{EXPERIMENT_CHECKPOINT_PATH}",
        language="text",
    )

model_col1, model_col2, model_col3 = st.columns(3)
with model_col1:
    st.markdown(
        info_card("Selected Architecture", "EfficientNet-B0"),
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
            f"Available ({EXPERIMENT_LABEL})"
            if model_status["available"]
            else "Not found",
        ),
        unsafe_allow_html=True,
    )

experiment_col1, experiment_col2 = st.columns(2)
with experiment_col1:
    st.markdown(
        info_card("Experiment", EXPERIMENT_LABEL), unsafe_allow_html=True
    )
with experiment_col2:
    st.markdown(
        info_card("Checkpoint", EXPERIMENT_CHECKPOINT_FILENAME),
        unsafe_allow_html=True,
    )

st.write(model_status["message"])
st.caption(
    "Experiment 1 is kept in the repository unchanged as a historical record. "
    "It is not the model this prototype loads."
)

# ---------------------------------------------------------------------------
# Section 4b - Experiment 2 training and final test results
# ---------------------------------------------------------------------------
st.markdown("#### Experiment 2 Training Summary")

training_summary = read_json_file(EXPERIMENT_TRAINING_SUMMARY_PATH)
evaluation_report = read_json_file(EXPERIMENT_EVALUATION_REPORT_PATH)

if training_summary is None:
    st.warning(
        "The Experiment 2 training summary file was not found, so no training "
        "figures are reported by this application."
    )
else:
    best_epoch = training_summary.get("best_epoch")
    epochs_completed = training_summary.get("epochs_completed")
    learning_rate = training_summary.get("learning_rate")
    stopped_early = training_summary.get("stopped_early")
    wall_clock_minutes = training_summary.get("wall_clock_minutes_this_run")

    train_col1, train_col2, train_col3, train_col4 = st.columns(4)
    with train_col1:
        st.metric("Epochs completed", epochs_completed)
    with train_col2:
        st.metric("Best epoch", best_epoch)
    with train_col3:
        st.metric("Learning rate", f"{learning_rate:.2e}")
    with train_col4:
        st.metric(
            "Early stopping",
            "No" if not stopped_early else "Yes",
        )

    st.caption(
        f"Training images: {training_summary.get('train_images')} | "
        f"Validation images: {training_summary.get('validation_images')} | "
        f"Learning rate: {learning_rate:.2e} | "
        f"Training duration: approximately "
        f"{wall_clock_minutes:.1f} minutes on CPU | "
        f"Dataset fingerprint: "
        f"{training_summary.get('dataset_fingerprint', 'unknown')}"
    )

if evaluation_report is None:
    st.warning(
        "The Experiment 2 evaluation report was not found, so no final test "
        "figures are reported by this application."
    )
else:
    metrics = evaluation_report.get("metrics", {})
    per_class = metrics.get("per_class", {})
    macro = metrics.get("macro", {})
    confusion_matrix = metrics.get("confusion_matrix", [])

    st.markdown("#### Experiment 2 Final Test Set Results")
    st.caption(
        f"Measured on the frozen {EXPERIMENT_LABEL} held-out test split of "
        f"{metrics.get('num_samples')} images "
        "(450 real + 450 fake). These are TEST results and are reported "
        "separately from the validation figures above."
    )

    result_col1, result_col2, result_col3 = st.columns(3)
    with result_col1:
        st.metric("Test images", metrics.get("num_samples"))
    with result_col2:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.2%}")
    with result_col3:
        st.metric("Macro F1", f"{macro.get('f1', 0.0):.2%}")

    st.markdown(
        f"- **Macro precision:** {macro.get('precision', 0.0):.2%}\n"
        f"- **Macro recall:** {macro.get('recall', 0.0):.2%}\n"
        f"- **Macro F1:** {macro.get('f1', 0.0):.2%}\n"
        f"- **Fake precision:** {per_class.get('fake', {}).get('precision', 0.0):.2%}"
        f" | **Fake recall:** {per_class.get('fake', {}).get('recall', 0.0):.2%}"
        f" | **Fake F1:** {per_class.get('fake', {}).get('f1', 0.0):.2%}\n"
        f"- **Real precision:** {per_class.get('real', {}).get('precision', 0.0):.2%}"
        f" | **Real recall:** {per_class.get('real', {}).get('recall', 0.0):.2%}"
        f" | **Real F1:** {per_class.get('real', {}).get('f1', 0.0):.2%}"
    )

    if len(confusion_matrix) == 2 and all(len(row) == 2 for row in confusion_matrix):
        true_positive_real = confusion_matrix[0][0]
        false_positive = confusion_matrix[0][1]
        false_negative = confusion_matrix[1][0]
        true_positive_fake = confusion_matrix[1][1]

        st.markdown(
            "**Confusion matrix (rows = true class, columns = predicted "
            f"class):** `{confusion_matrix}`\n\n"
            f"- Real → Real (correct): {true_positive_real}\n"
            f"- Real → Fake (false positive): {false_positive}\n"
            f"- Fake → Real (false negative): {false_negative}\n"
            f"- Fake → Fake (correct): {true_positive_fake}"
        )
        st.caption(
            f"False positives: {false_positive} | False negatives: "
            f"{false_negative}"
        )

    st.caption(
        "Source: `experiments/experiment_2_large_dataset/reports/"
        "evaluation_report.json`"
    )

    if training_summary is not None:
        st.info(
            "**Methodological note.** Experiment 2 achieved higher test "
            "performance under its specified larger-dataset and independently "
            "sourced data conditions, but the design does not permit "
            "attributing that difference to dataset size alone. Experiment 1 "
            "and Experiment 2 differ in multiple dimensions: dataset amount, "
            "fake-generation method/family, image framing, source "
            "resolution/downsampling characteristics, compression/file format "
            "and test-set size."
        )
        st.caption(
            "Validation metrics (best validation accuracy and best validation "
            "loss) describe model selection on the validation split only. They "
            "are not test accuracy."
        )

st.divider()


# ---------------------------------------------------------------------------
# Section 5 - Dataset status
# ---------------------------------------------------------------------------
st.subheader("5. Dataset Status")
st.write(
    f"This section reports the verified {EXPERIMENT_LABEL} dataset. The figures "
    "come from the committed Experiment 2 dataset build report and are "
    "cross-checked against the processed images on disk. Nothing here is "
    "estimated or invented."
)

dataset_report = read_json_file(EXPERIMENT_DATASET_REPORT_PATH)

if dataset_report is None:
    st.warning(
        "The Experiment 2 dataset report was not found, so no dataset figures "
        "are reported by this application."
    )
else:
    source = dataset_report.get("source", {})
    configuration = dataset_report.get("configuration", {})
    deduplication = dataset_report.get("deduplication", {})
    selected = dataset_report.get("selected", {})
    split_counts = selected.get("counts", {})
    min_cross_split = dataset_report.get("min_cross_split_phash_distance", {})

    selected_total = selected.get("total")
    selected_real = sum(
        split_counts.get(split_name, {}).get("real", 0)
        for split_name in ("train", "validation", "test")
    )
    selected_fake = sum(
        split_counts.get(split_name, {}).get("fake", 0)
        for split_name in ("train", "validation", "test")
    )
    selected_train = split_counts.get("train", {}).get("total")
    selected_validation = split_counts.get("validation", {}).get("total")
    selected_test = split_counts.get("test", {}).get("total")

    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        st.metric("Real images", f"{selected_real:,}")
    with status_col2:
        st.metric("Fake images", f"{selected_fake:,}")
    with status_col3:
        st.metric("Invalid / corrupt files", 0)

    status_col4, status_col5, status_col6 = st.columns(3)
    with status_col4:
        st.metric("Training images", f"{selected_train:,}")
    with status_col5:
        st.metric("Validation images", f"{selected_validation:,}")
    with status_col6:
        st.metric("Test images", f"{selected_test:,}")

    status_col7, status_col8, status_col9 = st.columns(3)
    with status_col7:
        st.metric("Total images", f"{selected_total:,}")
    with status_col8:
        st.metric(
            "Near-duplicates removed",
            f"{deduplication.get('near_duplicates_removed', 0):,}",
        )
    with status_col9:
        st.metric(
            "Exact duplicates",
            f"{deduplication.get('exact_duplicates_removed_cross_split', 0):,}",
        )

    st.success(f"✓ Verified {EXPERIMENT_LABEL} dataset found.")

    with st.expander("Split details"):
        for split_name in ("train", "validation", "test"):
            info = split_counts.get(split_name, {})
            st.write(
                f"**{split_name.capitalize()}:** "
                f"{info.get('real', 0):,} real + {info.get('fake', 0):,} fake "
                f"= {info.get('total', 0):,} images"
            )

    st.markdown("**Dataset preparation metadata**")
    st.markdown(
        f"- **Source dataset:** {source.get('dataset', 'unknown')}\n"
        f"- **Prepared dataset size:** {selected_total:,} images\n"
        f"- **Seed:** {configuration.get('seed', 'unknown')}\n"
        f"- **pHash threshold:** {configuration.get('phash_threshold', 'unknown')} bits\n"
        f"- **Near-duplicates removed:** "
        f"{deduplication.get('near_duplicates_removed', 0):,}\n"
        f"- **Exact duplicates:** "
        f"{deduplication.get('exact_duplicates_removed_same_class_same_split', 0):,}"
        f" same-class/same-split, "
        f"{deduplication.get('exact_duplicates_removed_cross_split', 0):,} "
        f"cross-split, "
        f"{deduplication.get('exact_duplicates_removed_cross_class', 0):,} "
        f"cross-class\n"
        f"- **Cross-split near-duplicate audit:** "
        f"{'Passed' if deduplication.get('no_near_duplicate_across_splits', dataset_report.get('audit', {}).get('no_near_duplicate_across_splits')) else 'Not passed'}\n"
        f"- **Minimum cross-split pHash distances:** "
        f"Train ↔ Validation: {min_cross_split.get('train|validation', 'n/a')} bits, "
        f"Train ↔ Test: {min_cross_split.get('train|test', 'n/a')} bits, "
        f"Validation ↔ Test: {min_cross_split.get('validation|test', 'n/a')} bits"
    )
    st.caption(
        f"Dataset fingerprint: "
        f"{training_summary.get('dataset_fingerprint', 'unknown') if training_summary else 'unknown'}"
        f" | Manifest SHA-256 (`dataset_manifest.csv`): "
        f"`374445e3a27e87942e17f6b0335dfc59dfc38afe37e1596f5b6232fd98c00922`"
    )

    with st.expander("Cross-check against the processed images on disk"):
        on_disk = dataset.get_processed_statistics(EXPERIMENT_PROCESSED_DIR)
        disk_total = sum(
            counts for split in on_disk.values() for counts in split.values()
        )
        st.write(
            f"**Images found in `experiments/experiment_2_large_dataset/"
            f"processed`:** {disk_total:,}"
        )
        for split_name in ("train", "validation", "test"):
            split_on_disk = on_disk.get(split_name, {})
            st.write(
                f"- **{split_name.capitalize()}:** "
                f"{split_on_disk.get('real', 0):,} real + "
                f"{split_on_disk.get('fake', 0):,} fake"
            )
        if disk_total == selected_total:
            st.success("✓ On-disk image count matches the verified dataset total.")
        else:
            st.warning(
                "The on-disk image count does not match the verified dataset "
                "total reported in the Experiment 2 dataset build report."
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
    "4. ✓ Prepare the Experiment 2 dataset (6,000 images from the 140k Real "
    "and Fake Faces dataset, with verified train/validation/test splits and "
    "near-duplicate auditing).\n"
    "5. ✓ Implement and run Experiment 2 model training "
    "(`experiments/experiment_2_large_dataset/train_experiment_2.py`).\n"
    "6. ✓ Validate the Experiment 2 trained model on the validation split.\n"
    "7. ✓ Complete the final Experiment 2 test evaluation using accuracy, "
    "precision, recall and F1-score (see "
    "`experiments/experiment_2_large_dataset/reports/evaluation_report.json`).\n"
    "8. ✓ Integrate the Experiment 2 trained model with DeepGuard.\n"
    "9. ✓ Implement single-image prediction.\n"
    "10. ✓ Perform unit and integration testing.\n"
    "11. Extend the system to additional media types if feasible "
    "(planned future work)."
)
st.caption(
    "Items 1-10 are complete. Item 11 (video and audio analysis) is planned "
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
    f"Atria Institute of Technology - Experiment 2 Prototype: "
    "Experiment 2 model integrated</div>",
    unsafe_allow_html=True,
)
