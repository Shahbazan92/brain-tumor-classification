
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
    BASE_DIR, "densenet201_brain_tumor_model.keras"
)
LABELS_PATH = os.path.join(BASE_DIR, "class_names.txt")

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Load class names
# -----------------------------
@st.cache_data
def load_class_names():
    if not os.path.exists(LABELS_PATH):
        return None

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]

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
# Streamlit interface
# -----------------------------
st.title("🧠 Brain Tumor Classification")
st.write(
    "Upload a brain MRI image to get a prediction "
    "from the trained DenseNet201 model."
)

try:
    model = load_model()


    class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

if model.output_shape[-1] != len(class_names):
    st.error(
        f"Model has {model.output_shape[-1]} output classes, "
        f"but we configured {len(class_names)} class names."
    )
    st.stop()

    if len(class_names) != model.output_shape[-1]:
        st.error(
            f"Your model has {model.output_shape[-1]} output classes, "
            f"but class_names.txt contains {len(class_names)} labels."
        )
        st.stop()

except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    if st.button("Classify Image", type="primary"):
        with st.spinner("Analyzing MRI image..."):

            try:
                input_height = model.input_shape[1]
                input_width = model.input_shape[2]

                image_resized = image.resize(
                    (input_width, input_height)
                )

                image_array = np.array(
                    image_resized,
                    dtype=np.float32
                )

                # IMPORTANT:
                # Adjust preprocessing to match your training code.
                # This currently assumes X was scaled to [0, 1].
                image_array = image_array / 255.0

                image_array = np.expand_dims(
                    image_array, axis=0
                )

                predictions = model.predict(
                    image_array, verbose=0
                )[0]

                predicted_index = int(np.argmax(predictions))
                predicted_class = class_names[predicted_index]
                confidence = float(predictions[predicted_index]) * 100

                st.success("Prediction completed!")

                st.subheader("Prediction Result")
                st.write(f"**Predicted Class:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2f}%")

                st.subheader("Class Probabilities")

                for index, probability in enumerate(predictions):
                    st.write(
                        f"{class_names[index]}: "
                        f"{probability * 100:.2f}%"
                    )

                    st.progress(float(probability))

            except Exception as error:
                st.error(f"Prediction failed: {error}")

st.divider()

st.caption(
    "Research and educational demonstration only. "
    "This AI prediction is not a medical diagnosis. "
    "Always consult a qualified medical professional."
)
