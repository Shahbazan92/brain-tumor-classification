
import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "densenet201_brain_tumor_model.keras"
)

# Four brain tumor classes
# IMPORTANT: These must match the original training class order.
CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# -----------------------------
# Application interface
# -----------------------------
st.title("🧠 Brain Tumor Classification")

st.write(
    "Upload a brain MRI image to get a prediction "
    "from the trained DenseNet201 model."
)

# Load and validate model
try:
    model = load_model()

    output_classes = model.output_shape[-1]

    if output_classes != len(CLASS_NAMES):
        st.error(
            f"Model has {output_classes} output classes, "
            f"but we configured {len(CLASS_NAMES)} class names."
        )
        st.stop()

    st.success("DenseNet201 model loaded successfully!")

except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose a brain MRI image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Image prediction
# -----------------------------
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded MRI Image",
            use_container_width=True
        )

        if st.button("Classify Image", type="primary"):

            with st.spinner("Analyzing MRI image..."):

                input_height = model.input_shape[1]
                input_width = model.input_shape[2]

                image_resized = image.resize(
                    (input_width, input_height)
                )

                image_array = np.array(
                    image_resized,
                    dtype=np.float32
                )

                # Assumes training images were scaled to [0, 1].
                # Verify this against your original training pipeline.
                image_array = image_array / 255.0

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                # Generate predictions
                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(predictions)
                )

                predicted_class = CLASS_NAMES[predicted_index]

                confidence = (
                    float(predictions[predicted_index]) * 100
                )

                # Display result
                st.success("Prediction completed!")

                st.subheader("Prediction Result")

                st.write(
                    f"**Predicted Class:** {predicted_class}"
                )

                st.write(
                    f"**Confidence:** {confidence:.2f}%"
                )

                # Display probabilities
                st.subheader("Class Probabilities")

                for index, probability in enumerate(predictions):

                    st.write(
                        f"{CLASS_NAMES[index]}: "
                        f"{probability * 100:.2f}%"
                    )

                    st.progress(float(probability))

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# -----------------------------
# Medical disclaimer
# -----------------------------
st.divider()

st.caption(
    "Research and educational demonstration only. "
    "This AI prediction is not a medical diagnosis. "
    "Always consult a qualified medical professional."
)
