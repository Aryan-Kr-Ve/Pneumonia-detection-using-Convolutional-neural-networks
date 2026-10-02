import os

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
MODEL_CANDIDATES = ["trained.h5", "model/trained.h5"]   # first one found is used
CLASS_NAMES = ["Normal", "Pneumonia"]   # VERIFY: index 0 -> Normal, index 1 -> Pneumonia
PIXEL_SCALE = 255.0                     # VERIFY: pixels divided by this (255.0 = rescale 1/255)
THRESHOLD = 0.5                         # used only if the model has a single sigmoid output
TRAINING_NOTES = "See the training notebook (Pneumonia_detection_using_CNN.ipynb) for epochs, dataset and accuracy."

DISCLAIMER = (
    "**Medical Disclaimer:** This application is for educational and research purposes only. "
    "It is not a medical diagnostic tool and should not be used to diagnose, treat, or rule out "
    "pneumonia. AI predictions can be incorrect. Please consult a qualified healthcare "
    "professional for proper medical evaluation."
)

st.set_page_config(page_title="Pneumonia Detection Using CNN", page_icon="🫁", layout="wide")


# ---------------------------------------------------------------------------
# Model loading (cached: runs once per server session, never retrains)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("Model file not found: " + " or ".join(MODEL_CANDIDATES))
    from tensorflow.keras.models import load_model as keras_load
    return keras_load(path, compile=False)   # compile=False: inference only


def model_io(model):
    """Read input/output details from the model instead of guessing."""
    shape = model.input_shape                # e.g. (None, H, W, C)
    if isinstance(shape, list):
        shape = shape[0]
    return {"height": shape[1], "width": shape[2], "channels": shape[3],
            "outputs": int(model.output_shape[-1])}


# ---------------------------------------------------------------------------
# Preprocessing: resize -> colour mode -> scale -> add batch dimension
# ---------------------------------------------------------------------------
def preprocess_image(image: Image.Image, info: dict) -> np.ndarray:
    image = image.convert("L" if info["channels"] == 1 else "RGB")
    image = image.resize((info["width"], info["height"]))
    arr = np.asarray(image, dtype="float32") / PIXEL_SCALE
    if info["channels"] == 1:
        arr = np.expand_dims(arr, axis=-1)
    return np.expand_dims(arr, axis=0)       # shape: (1, H, W, C)


# ---------------------------------------------------------------------------
# Prediction: returns probability per class, in CLASS_NAMES order
# ---------------------------------------------------------------------------
def predict_image(model, batch: np.ndarray, info: dict) -> dict:
    out = np.asarray(model.predict(batch, verbose=0))[0]
    if info["outputs"] == 1:                 # single sigmoid: P(class index 1)
        p1 = float(out[0])
        probs = [1.0 - p1, p1]
        idx = 1 if p1 >= THRESHOLD else 0
    else:                                    # softmax over several classes
        probs = [float(x) for x in out]
        idx = int(np.argmax(probs))
    return {"label": CLASS_NAMES[idx], "confidence": probs[idx],
            "probs": dict(zip(CLASS_NAMES, probs))}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def display_prediction(result: dict):
    is_pneu = result["label"].lower().startswith("pneu")
    if is_pneu:
        st.error("### Prediction: Pneumonia Detected")
    else:
        st.success("### Prediction: Normal / No Pneumonia Detected")

    st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
    st.progress(min(max(result["confidence"], 0.0), 1.0))

    st.markdown("| Class | Probability |\n|---|---:|\n" + "\n".join(
        f"| {k} | {v * 100:.2f}% |" for k, v in result["probs"].items()))
    st.caption("This is the model's estimate, not a diagnosis.")

    st.subheader("What this means")
    if is_pneu:
        st.write(
            "**The model flagged patterns it associates with pneumonia.** This is not a diagnosis.\n\n"
            "- **What it is:** an infection that inflames the air sacs (alveoli) in one or both lungs.\n"
            "- **Common causes:** bacteria, viruses (including flu and COVID-19), and less often fungi or inhaled material.\n"
            "- **Common symptoms:** cough (sometimes with phlegm), fever, chills, shortness of breath, chest pain when breathing or coughing, fatigue.\n"
            "- **Why chest X-rays are used:** they can show areas of the lung that look cloudy or filled with fluid.\n"
            "- **Why confirmation is needed:** other conditions can look similar on an X-ray, and a clinician also considers symptoms, exam findings and tests.\n"
            "- **Seek medical attention** if you have a persistent cough or fever, trouble breathing, chest pain, or confusion. "
            "Get urgent care for severe breathlessness, bluish lips, or symptoms that are quickly worsening. "
            "Infants, older adults and people with weak immune systems should be seen sooner."
        )
    else:
        st.write(
            "**The model did not find patterns it associates with pneumonia in this image.**\n\n"
            "- A \"normal\" prediction does **not** guarantee that pneumonia or any other condition is absent.\n"
            "- The model can miss cases, and it only looks at one image, without your symptoms or history.\n"
            "- If you have symptoms such as cough, fever or breathing difficulty, see a healthcare professional regardless of this result."
        )


def show_about(info):
    with st.expander("About the model"):
        st.markdown(
            f"""
- **CNN:** a neural network that learns visual features (edges, textures, shapes) using convolution filters, then classifies them.
- **Model file:** `trained.h5` (Keras/TensorFlow), loaded once and cached.
- **Input image:** {info['width']} x {info['height']} pixels, {'grayscale' if info['channels'] == 1 else 'RGB (3 channels)'} (read from the model).
- **Preprocessing:** resize, convert colour mode, divide pixel values by {PIXEL_SCALE:g}, add a batch dimension.
- **Output:** {info['outputs']} unit(s) -> {'one sigmoid probability, threshold ' + str(THRESHOLD) if info['outputs'] == 1 else 'probability for each class'}.
- **Classes:** {', '.join(f'{i} = {c}' for i, c in enumerate(CLASS_NAMES))}
- **Training info:** {TRAINING_NOTES}
- **Limitations:** trained on a limited dataset; may not generalise to other hospitals, scanners or age groups; can give false positives and false negatives; not clinically validated.
"""
        )


def show_pipeline():
    with st.expander("How it works"):
        st.code("Upload X-Ray\n      ↓\nImage Preprocessing\n      ↓\nCNN Model\n      ↓\nFeature Extraction\n      ↓\nClassification\n      ↓\nPrediction Probability\n      ↓\nResult Display", language="text")
        st.markdown(
            "1. **Upload:** you provide a JPG/PNG chest X-ray.\n"
            "2. **Preprocessing:** the image is resized and scaled the same way as during training.\n"
            "3. **CNN:** the network processes the image.\n"
            "4. **Feature extraction:** convolution layers pick up patterns such as edges and textures.\n"
            "5. **Classification:** final layers turn those features into a score.\n"
            "6. **Probability:** the score is read as the chance of each class.\n"
            "7. **Result:** the most likely class and its confidence are shown."
        )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Pneumonia Detection")
    st.subheader("Upload X-ray")
    st.write("Use the uploader on the main page (JPG, JPEG or PNG).")
    st.subheader("How it works")
    st.write("Your image is preprocessed and passed to a trained CNN, which outputs class probabilities.")
    st.subheader("Model information")
    st.write("Keras CNN loaded from `trained.h5`. Details are under *About the model*.")
    st.subheader("Medical disclaimer")
    st.warning(DISCLAIMER)

st.title("Pneumonia Detection Using CNN")
st.write("Upload a chest X-ray and a trained Convolutional Neural Network will estimate whether it shows patterns associated with pneumonia.")
st.warning(DISCLAIMER)

try:
    model = load_model()
    info = model_io(model)
except FileNotFoundError as e:
    st.error(f"{e}. Place your trained model next to app.py and restart.")
    st.stop()
except Exception:
    st.error("The model could not be loaded. Check that TensorFlow is installed and the model file is not corrupted (see README > Troubleshooting).")
    st.stop()

uploaded = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
col_img, col_res = st.columns(2)

if uploaded is not None:
    try:
        image = Image.open(uploaded)
        image.load()   # forces decoding so corrupted files fail here
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be read as an image. Please upload a valid JPG or PNG.")
        st.stop()

    with col_img:
        st.subheader("Uploaded X-ray")
        st.image(image, use_container_width=True)

    with col_res:
        st.subheader("Result")
        if st.button("Analyze X-Ray", type="primary"):
            try:
                with st.spinner("Analyzing..."):
                    batch = preprocess_image(image, info)
                    result = predict_image(model, batch, info)
                display_prediction(result)
            except Exception:
                st.error("Prediction failed. Try a different image or check the model configuration.")
        else:
            st.info("Click **Analyze X-Ray** to run the model.")
else:
    st.info("Please upload an X-ray image to begin.")

show_about(info)
show_pipeline()
