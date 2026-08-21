import streamlit as st
import tensorflow as tf
import numpy as np
import json
import io

from PIL import Image
from gradcam import make_gradcam_heatmap


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Naksh AI | Telescope Assistant",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS ONLY
# No HTML divs are used in the interface.
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- MAIN APP ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 15%,
                rgba(75, 0, 130, 0.30),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(0, 120, 255, 0.22),
                transparent 30%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(120, 0, 255, 0.20),
                transparent 35%
            ),
            #050817;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }


    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #080b24 0%,
                #111538 50%,
                #070918 100%
            );

        border-right:
            1px solid rgba(255,255,255,0.12);
    }


    /* ---------- NATIVE TITLES ---------- */

    h1 {
        color: #f8fafc !important;
    }

    h2 {
        color: #e2e8f0 !important;
    }

    h3 {
        color: #cbd5e1 !important;
    }

    p {
        color: #cbd5e1;
    }


    /* ---------- HERO ---------- */

    .hero-title {
        font-size: 3.8rem;
        font-weight: 900;
        text-align: center;
        letter-spacing: 0.12em;

        background:
            linear-gradient(
                90deg,
                #67e8f9,
                #818cf8,
                #c084fc,
                #f0abfc
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        margin-bottom: 0.2rem;

        text-shadow:
            0 0 30px rgba(129,140,248,0.35);
    }

    .hero-subtitle {
        text-align: center;
        font-size: 1.15rem;
        color: #94a3b8;
        letter-spacing: 0.08em;
        margin-bottom: 1.2rem;
    }


    /* ---------- HERO DIVIDER ---------- */

    .hero-line {
        height: 2px;

        background:
            linear-gradient(
                90deg,
                transparent,
                #6366f1,
                #c084fc,
                #22d3ee,
                transparent
            );

        margin: 0.5rem 10% 1.5rem 10%;
    }


    /* ---------- GLASS CONTAINERS ---------- */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background:
            rgba(255,255,255,0.045);

        border:
            1px solid rgba(255,255,255,0.10);

        border-radius: 18px;
    }


    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 12px;

        border:
            1px solid rgba(129,140,248,0.45);

        background:
            linear-gradient(
                90deg,
                #4f46e5,
                #7c3aed
            );

        color: white;

        font-weight: 700;

        min-height: 48px;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 0 25px
            rgba(124,58,237,0.45);
    }


    /* ---------- FILE UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        background:
            rgba(99,102,241,0.07);

        border:
            1px dashed rgba(129,140,248,0.45);

        border-radius: 18px;

        padding: 8px;
    }


    /* ---------- METRICS ---------- */

    [data-testid="stMetric"] {
        background:
            rgba(255,255,255,0.045);

        border:
            1px solid rgba(255,255,255,0.10);

        border-radius: 16px;

        padding: 15px;
    }


    /* ---------- TABS ---------- */

    button[data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 700;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #c4b5fd;
    }


    /* ---------- PROGRESS ---------- */

    [data-testid="stProgressBar"] {
        border-radius: 10px;
    }


    /* ---------- FOOTER ---------- */

    .footer-text {
        text-align: center;
        color: #64748b;
        padding: 30px 0 10px 0;
    }


    /* ---------- ANIMATION ---------- */

    @keyframes pulseGlow {
        0% {
            opacity: 0.75;
        }

        50% {
            opacity: 1;
        }

        100% {
            opacity: 0.75;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    '<div class="hero-title">🔭 NAKSH AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'AI-POWERED ASTRONOMICAL IMAGE ASSISTANT'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-line"></div>',
    unsafe_allow_html=True
)

# ============================================================
# HERO INTRODUCTION
# ============================================================

st.markdown(
    "### Explore the Universe with AI"
)

st.write(
    "Upload an astronomical image and let "
    "Naksh AI identify the celestial object."
)

st.write("")

intro1, intro2, intro3 = st.columns(3)

with intro1:

    st.markdown(
        "### Upload"
    )

    st.caption(
        "Choose an astronomical image "
        "from your device."
    )

with intro2:

    st.markdown(
        "### Analyze"
    )

    st.caption(
        "Our AI analyzes visual features "
        "in the image."
    )

with intro3:

    st.markdown(
        "### Discover"
    )

    st.caption(
        "Explore the model's prediction "
        "and visual explanation."
    )


# ============================================================
# SUPPORTED OBJECTS
# ============================================================

st.write("")

object_columns = st.columns(4)

objects = [
    ("🌙", "MOON", "Natural satellite"),
    ("🌌", "NEBULA", "Interstellar cloud"),
    ("🪐", "PLANET", "Planetary body"),
    ("⭐", "STAR", "Stellar object")
]


for column, item in zip(
    object_columns,
    objects
):

    icon, name, description = item

    with column:

        with st.container(
            border=True
        ):

            st.markdown(
                f"## {icon}"
            )

            st.markdown(
                f"**{name}**"
            )

            st.caption(
                description
            )


st.write("")


# ============================================================
# SIDEBAR
# ============================================================
# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Naksh AI")

    st.caption(
        "Astronomical Image Assistant"
    )

    st.divider()

    st.subheader(
        "Explore"
    )

    st.write("Image Classification")
    st.write("AI Explainability")
    st.write("Prediction Analysis")

    st.divider()

    st.subheader(
        "Recognized Objects"
    )

    st.write("Moon")
    st.write("Nebula")
    st.write("Planet")
    st.write("Star")

    st.divider()

    st.subheader(
        "AI Technology"
    )

    st.write("TensorFlow / Keras")
    st.write("MobileNetV3Small")
    st.write("Transfer Learning")
    st.write("Grad-CAM")

    st.divider()

    st.caption(
        "Upload a clear astronomical image "
        "for the best experience."
    )

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "model/best_model.keras"
    )


@st.cache_data
def load_class_names():

    with open(
        "model/class_names.json",
        "r"
    ) as file:

        return json.load(file)


try:

    model = load_model()

    class_names = load_class_names()

except Exception as e:

    st.error(
        "Could not load the Naksh AI model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "CLASSIFY",
        "PERFORMANCE",
        "ABOUT"
    ]
)


# ============================================================
# CLASSIFICATION TAB
# ============================================================

with tab1:

    st.header(
        "Astronomical Image Analysis"
    )

    st.write(
        "Upload an image from a telescope or "
        "astronomical dataset."
    )


    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        st.divider()


        # ----------------------------------------------------
        # IMAGE PREVIEW
        # ----------------------------------------------------

        image_col, info_col = st.columns(
            [1.2, 1]
        )


        with image_col:

            st.image(
                image,
                caption="Uploaded astronomical image",
                use_container_width=True
            )


        with info_col:

            st.subheader(
                "Image Information"
            )

            st.write(
                f"**File:** {uploaded_file.name}"
            )

            st.write(
                f"**Resolution:** "
                f"{image.width} × {image.height}"
            )

            st.write(
                "**Color:** RGB"
            )

            st.write(
                "**AI Input:** 224 × 224"
            )

            st.info(
                "Naksh AI will resize the image "
                "before classification."
            )


        st.write("")


        # ----------------------------------------------------
        # ANALYZE
        # ----------------------------------------------------

        analyze = st.button(
            "ANALYZE WITH NAKSH AI",
            type="primary",
            use_container_width=True
        )


        if analyze:

            with st.spinner(
                "🔭 Analyzing the cosmos..."
            ):

                # ==========================================
                # PREPROCESS
                # ==========================================

                resized = image.resize(
                    (224, 224)
                )

                image_array = np.array(
                    resized
                ).astype(
                    np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )


                # ==========================================
                # PREDICTION
                # ==========================================

                predictions = model.predict(
                    image_array,
                    verbose=0
                )

                predicted_index = int(
                    np.argmax(
                        predictions[0]
                    )
                )

                predicted_class = class_names[
                predicted_index
                ]

                confidence = float(
                    predictions[0][
                       predicted_index
                    ]
                )

                # ============================================================
                # UNKNOWN IMAGE REJECTION
                # ============================================================

                UNKNOWN_THRESHOLD = 0.75

                if confidence < UNKNOWN_THRESHOLD:
                    is_unknown = True
                else:
                    is_unknown = False


                # ==========================================
                # GRAD-CAM
                # ==========================================

                heatmap, _ = (
                    make_gradcam_heatmap(
                        image,
                        model,
                        "conv_1"
                    )
                )


            # =================================================
            # RESULT
            # =================================================

            st.divider()

            st.header(
                "Detection Result"
            )


            result1, result2 = st.columns(
                2
            )

            # ============================================================
            # DETECTION RESULT
            # ============================================================

            if is_unknown:

                st.error(
                    "UNKNOWN / UNRECOGNIZED IMAGE"
                )

                st.warning(
                    f"""
                    Naksh AI could not confidently identify
                    this image as a Moon, Nebula, Planet, or Star.

                    Highest prediction:
                    **{predicted_class.upper()}**
                    
                    Confidence:
                    **{confidence * 100:.2f}%**
                    
                    Please upload a clearer astronomical image.
                    """
                )

            else:

                result1, result2 = st.columns(2)

                with result1:

                    st.metric(
                        "Detected Object",
                        predicted_class.upper()
                    )

                with result2:

                    st.metric(
                        "AI Confidence",
                        f"{confidence * 100:.2f}%"
                    )


                if confidence >= 0.90:

                    st.success(
                        f"High confidence detection — "
                        f"{predicted_class.upper()}"
                    )

                elif confidence >= 0.75:

                    st.warning(
                        f"Moderate confidence — "
                        f"{predicted_class.upper()}"
                    )

            # =================================================
            # PROBABILITIES
            # =================================================

            st.subheader(
                "📊 AI Probability Distribution"
            )


            for i, name in enumerate(
                class_names
            ):

                probability = float(
                    predictions[0][i]
                )

                st.write(
                    f"**{name.capitalize()}** "
                    f"— {probability * 100:.2f}%"
                )

                st.progress(
                    probability
                )


            # =================================================
            # GRAD-CAM
            # =================================================

            if not is_unknown:
                st.divider()

                st.header(
                    "Explainable AI"
                )

                st.write(
                    "Grad-CAM shows which regions "
                    "influenced the model's decision."
                )


            # -------------------------------------------------
            # HEATMAP
            # -------------------------------------------------

            heatmap_image = Image.fromarray(
                np.uint8(
                    heatmap * 255
                )
            )

            heatmap_image = heatmap_image.resize(
                image.size,
                Image.Resampling.BILINEAR
            )

            heatmap_array = (
                np.array(
                    heatmap_image
                ) / 255.0
            )


            # -------------------------------------------------
            # COLOUR MAP
            # -------------------------------------------------

            import matplotlib.pyplot as plt

            cmap = plt.get_cmap(
                "jet"
            )

            colored_heatmap = cmap(
                heatmap_array
            )

            colored_heatmap = np.uint8(
                colored_heatmap[:, :, :3]
                * 255
            )


            # -------------------------------------------------
            # OVERLAY
            # -------------------------------------------------

            original_array = np.array(
                image
            )

            overlay = (
                0.55 * original_array
                +
                0.45 * colored_heatmap
            )

            overlay = np.uint8(
                np.clip(
                    overlay,
                    0,
                    255
                )
            )


            # -------------------------------------------------
            # DISPLAY
            # -------------------------------------------------

            cam1, cam2, cam3 = st.columns(
                3
            )


            with cam1:

                st.image(
                    image,
                    caption="Original",
                    use_container_width=True
                )


            with cam2:

                st.image(
                    colored_heatmap,
                    caption="Grad-CAM",
                    use_container_width=True
                )


            with cam3:

                st.image(
                    overlay,
                    caption="AI Focus",
                    use_container_width=True
                )


            # =================================================
            # DOWNLOAD
            # =================================================

            buffer = io.BytesIO()

            Image.fromarray(
                overlay
            ).save(
                buffer,
                format="PNG"
            )

            st.download_button(
                "Download AI Focus Image",
                buffer.getvalue(),
                "naksh_ai_gradcam.png",
                "image/png",
                use_container_width=True
            )


            # =================================================
            # EXPLANATION
            # =================================================

            st.divider()

            st.subheader(
                "Understanding the Result"
            )

            st.info(
                f"""
                Naksh AI classified the image as
                **{predicted_class.upper()}**
                with **{confidence * 100:.2f}% confidence**.

                🔴 Red / yellow regions indicate stronger
                influence on the prediction.

                🔵 Blue regions indicate weaker influence.

                Grad-CAM explains model attention. It is
                not an exact scientific boundary.
                """
            )


# ============================================================
# PERFORMANCE TAB
# ============================================================

with tab2:

    st.header(
        "Model Performance"
    )

    st.write(
        "Performance measured on the held-out test dataset."
    )


    metric1, metric2, metric3 = st.columns(3)


    with metric1:

        st.metric(
            "Test Accuracy",
            "98.26%"
        )


    with metric2:

        st.metric(
            "Test Images",
            "690"
        )


    with metric3:

        st.metric(
            "Classes",
            "4"
        )


    st.divider()


    st.subheader(
        "Class-wise Performance"
    )


    performance = {
        "Moon": (98.02, 94.29, 96.12),
        "Nebula": (97.92, 96.58, 97.24),
        "Planet": (97.32, 99.61, 98.45),
        "Star": (100.00, 100.00, 100.00)
    }


    for name, values in performance.items():

        st.write(
            f"### {name}"
        )

        p1, p2, p3 = st.columns(3)

        with p1:

            st.metric(
                "Precision",
                f"{values[0]:.2f}%"
            )

        with p2:

            st.metric(
                "Recall",
                f"{values[1]:.2f}%"
            )

        with p3:

            st.metric(
                "F1 Score",
                f"{values[2]:.2f}%"
            )


    st.divider()

    st.subheader(
        "Confusion Matrix"
    )


    try:

        st.image(
            "model/confusion_matrix.png",
            caption="Test-set confusion matrix",
            use_container_width=True
        )

    except Exception:

        st.warning(
            "Confusion matrix image not found."
        )


# ============================================================
# ABOUT TAB
# ============================================================

with tab3:

    st.header(
        "About Naksh AI"
    )

    st.write(
        """
        **Naksh AI** is an AI-powered telescope assistant
        designed to classify astronomical images and provide
        an interpretable prediction.
        """
    )


    st.divider()


    st.subheader(
        "How Naksh AI Works"
    )


    step1, step2 = st.columns(2)


    with step1:

        st.markdown(
            """
            ### 01  Upload

            Upload a telescope or astronomical image.

            ### 02 Preprocess

            The image is resized to 224 × 224 pixels.

            ### 03 Feature Extraction

            MobileNetV3Small extracts visual features.
            """
        )


    with step2:

        st.markdown(
            """
            ### 04 Classify

            The model predicts one of four classes.

            ### 05 Explain

            Grad-CAM highlights influential regions.

            ### 06 Analyze

            View confidence and class probabilities.
            """
        )


    st.divider()


    st.subheader(
        "Technology Stack"
    )


    tech1, tech2, tech3, tech4 = st.columns(4)


    with tech1:

        st.metric(
            "🐍",
            "Python"
        )


    with tech2:

        st.metric(
            "🧠",
            "TensorFlow"
        )


    with tech3:

        st.metric(
            "🌐",
            "Streamlit"
        )


    with tech4:

        st.metric(
            "🔥",
            "Grad-CAM"
        )


    st.divider()


    st.subheader(
        "Future Scope"
    )

    st.write(
        """
        • More astronomical classes

        • Larger and more diverse datasets

        • Real telescope camera integration

        • Real-time classification

        • Astronomical metadata

        • Object detection

        • Satellite and space-object recognition

        • Unknown-object rejection

        • Cloud deployment
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Naksh AI • AI Telescope Assistant • "
    "Exploring the universe through Artificial Intelligence "
)
