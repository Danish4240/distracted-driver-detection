import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import albumentations as A
from albumentations.pytorch import ToTensorV2
import google.generativeai as genai

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.models import build_model

# Streamlit Page Config
st.set_page_config(
    page_title="Distracted Driver Detection & AI Safety Coach",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Distracted Driver Detection System")
st.markdown("Upload a driver camera snapshot to analyze driving state, check seatbelt compliance, and receive GenAI safety guidance.")

# --- Model Loading ---
@st.cache_resource
def load_detection_model():
    weights_path = os.path.join(Config.WEIGHTS_DIR, "best_model.pth")
    model = build_model(model_name="efficientnet_b0", pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=Config.DEVICE))
    model.eval()
    return model

def get_inference_transform():
    return A.Compose([
        A.Resize(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# Sidebar Configuration
st.sidebar.header("Configuration")

# Retrieve API key from Streamlit Secrets or Environment Variables automatically
default_api_key = st.secrets.get("GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

gemini_api_key = st.sidebar.text_input(
    "Gemini API Key (for Seatbelt & AI Coach)",
    value=default_api_key,
    type="password"
)

try:
    model = load_detection_model()
    st.sidebar.success("EfficientNet-B0 weights loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model weights: {e}")

threshold = st.sidebar.slider("Alert Confidence Threshold", 0.0, 1.0, 0.5)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded Driver Snapshot", use_container_width=True)

    # Transform image for PyTorch Model
    transform = get_inference_transform()
    img_np = np.array(image)
    augmented = transform(image=img_np)
    tensor_img = augmented["image"].unsqueeze(0).to(Config.DEVICE)

    # Run PyTorch Model Prediction
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = F.softmax(outputs, dim=1)[0]
        top_prob, top_catid = torch.max(probabilities, 0)
        predicted_label = Config.CLASS_LABELS[f"c{top_catid.item()}"]
        confidence = top_prob.item()

    with col2:
        st.subheader("Inference Diagnostics")
        if predicted_label == "Safe Driving":
            st.success(f"**State:** {predicted_label}")
        else:
            st.error(f"**Distraction Detected:** {predicted_label}")

        st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")

        if confidence < threshold:
            st.warning("Confidence is below target threshold. Verification advised.")

    # --- GenAI Seatbelt & Safety Coaching Integration ---
    st.markdown("---")
    st.subheader("🤖 GenAI Multimodal Analysis")

    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
            prompt = f"""
            Analyze this driver snapshot image and answer two points briefly:
            1. Seatbelt Status: Is the driver wearing a seatbelt properly across their torso/shoulder? Answer with 'Seatbelt Detected', 'No Seatbelt Detected', or 'Unclear'.
            2. AI Safety Coaching: The model detected the driver behavior as '{predicted_label}'. Write a polite, concise 1-sentence alert/coaching statement for the driver.
            """
            with st.spinner("Analyzing seatbelt & generating safety prompt via Gemini API..."):
                response = gemini_model.generate_content([prompt, image])
                st.info(response.text)
        except Exception as e:
            st.error(f"GenAI Analysis failed: {e}")
    else:
        st.warning("Enter your Gemini API Key in the left sidebar to enable real-time Seatbelt Detection & GenAI Coaching alerts.")

    # --- Matplotlib Chart with 45-degree Rotated Labels ---
    st.markdown("---")
    st.subheader("Class Probabilities Distribution")
    labels = [Config.CLASS_LABELS[f"c{i}"] for i in range(10)]
    probs = [float(probabilities[i]) for i in range(10)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, probs, color="#1f77b4")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.0)

    # Rotate x-axis labels by 45 degrees for better readability
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    st.pyplot(fig)
