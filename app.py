import json
import time
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from config import *

# Page configuration
st.set_page_config(
    page_title="Bhutan Bird AI",
    page_icon="🐦",
    layout="wide"
)

st.title("🐦 Bhutan Bird AI")
st.caption("MLA306 • Bhutanese Bird Species Classification")

# Select the best-performing model
MODEL_PATH = MODELS / "efficientnet.keras"


# Load trained model and class names
@st.cache_resource
def load():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    names = json.loads(
        (DATA / "classes.json").read_text()
    )

    return model, names


# Check whether the model exists
if not MODEL_PATH.exists():
    st.warning("EfficientNet model not found.")
    st.stop()


m, names = load()

st.sidebar.header("Model Information")
st.sidebar.write("Model: EfficientNetB0")
st.sidebar.write("Supported species: 11")
st.sidebar.write("Training images: 1,100")
st.sidebar.write("Testing images: 55")
st.sidebar.write("Test accuracy: 100%")

st.sidebar.caption(
    "Accuracy is based on the selected test dataset, "
    "not all real-world bird photographs."
)

# Display supported species
with st.sidebar.expander("Supported Bird Species"):
    for name in names:
        st.write(name)


# Upload image
f = st.file_uploader(
    "Upload a bird photograph",
    type=["jpg", "jpeg", "png"]
)

if f:

    # Image preprocessing
    im = Image.open(f).convert("RGB")

    x = np.asarray(
        im.resize(SIZE),
        dtype=np.float32
    )[None]

    start = time.perf_counter()

    # Make prediction
    p = m(x, training=False)

    idx = int(tf.argmax(p[0]))

    confidence = float(p[0, idx]) * 100

    # Display prediction
    st.subheader("Classification Result")

    c1, c2 = st.columns(2)

    c1.metric(
        "Predicted Species",
        names[idx].title()
    )

    c2.metric(
        "Model Confidence",
        f"{confidence:.2f}%"
    )

    if confidence < 60:
        st.warning(
            "Low confidence. Try uploading a clearer "
            "photograph of the bird."
        )

    # Original image and Grad-CAM
    st.subheader("Explainable AI — Grad-CAM")

    col1, col2 = st.columns(2)

    col1.image(
        im,
        caption="Original Bird Image",
        use_container_width=True
    )

    try:

        grad_model = tf.keras.Model(
            m.inputs,
            [
                m.get_layer("cam_features").output,
                m.output
            ]
        )

        with tf.GradientTape() as tape:

            features, predictions = grad_model(
                x,
                training=False
            )

            score = predictions[:, idx]

        grads = tape.gradient(score, features)

        weights = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        heat = tf.maximum(
            tf.reduce_sum(
                features[0] * weights,
                axis=-1
            ),
            0
        )

        heat = heat / (
            tf.reduce_max(heat) + 1e-8
        )

        heat = tf.image.resize(
            heat[..., None],
            SIZE
        ).numpy()[..., 0]

        rgb = np.asarray(
            im.resize(SIZE)
        ) / 255.0

        overlay = np.clip(
            0.55 * rgb
            + 0.45 * plt.get_cmap("jet")(heat)[..., :3],
            0,
            1
        )

        col2.image(
            overlay,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

        st.caption(
            "The heatmap highlights image regions "
            "that contributed to the model's prediction."
        )

    except Exception as e:

        col2.warning("Grad-CAM could not be generated.")
        st.caption(str(e))

    # Top three predictions
    st.subheader("Top 3 Predictions")

    top3 = np.argsort(p[0].numpy())[::-1][:3]

    for i in top3:

        probability = float(p[0, i])

        st.write(
            f"**{names[i].title()} — "
            f"{probability * 100:.2f}%**"
        )

        st.progress(probability)

    # Processing time
    elapsed = time.perf_counter() - start

    st.caption(
        f"Inference and explanation time: "
        f"{elapsed:.2f} seconds"
    )

    st.warning(
        "This model supports only 11 selected bird "
        "species. Predictions and confidence scores "
        "are not guarantees of correctness."
    )

else:

    st.info(
        "Upload a bird photograph to identify its species."
    )