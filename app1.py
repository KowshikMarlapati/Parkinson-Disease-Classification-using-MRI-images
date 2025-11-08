import streamlit as st
import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
import tempfile

# ==== CONFIG ====
MODEL_PATH = Path(r"C:\Users\kowsh\OneDrive\Desktop\ViT\_CAPSTONE_\Parkinson Disease\cnn_direct_training_output\mobilenetv2_mri_pd_classifier.keras")
IMG_SIZE = (224, 224)
THRESHOLD = 0.734

# ==== Load Model ====
@st.cache_resource
def load_pd_model():
    return load_model(MODEL_PATH)

model = load_pd_model()

# ==== Preprocess Function ====
def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img).astype("float32")
    img = preprocess_input(img)
    return img

def extract_patient_id(fname):
    if "patient___" in fname:
        return fname.split("patient___")[1].split("___")[0]
    return "Unknown"

# ==== UI Title ====
st.title(" MRI Parkinson Disease Classification")
st.write("Upload MRI slices and click **Classify** to analyze for Parkinson's vs Healthy.")

uploaded_files = st.file_uploader(
    "Upload MRI slices (PNG/JPG)...",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg"]
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} images uploaded ✅")

    # Temporary folder to store uploads
    temp_dir = tempfile.TemporaryDirectory()
    temp_folder = Path(temp_dir.name)

    image_paths = []
    for file in uploaded_files:
        file_path = temp_folder / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        image_paths.append(str(file_path))

    # ✅ Classify button
    if st.button(" Classify"):
        with st.spinner("Running classification... Please wait ⏳"):
            # Preprocess images
            images = np.stack([preprocess_image(p) for p in image_paths])

            # Predict
            pred_probs = model.predict(images, verbose=0).flatten()
            pred_labels = (pred_probs >= THRESHOLD).astype(int)

            df = pd.DataFrame({
                "image": [os.path.basename(p) for p in image_paths],
                "prob_PD": pred_probs,
                "prediction": ["PD" if i == 1 else "Healthy" for i in pred_labels]
            })

            # Slice-level results
            st.subheader("🔍 Slice-Level Classification")
            st.dataframe(df)

            # ✅ Patient-level Aggregation
            df["patient"] = df["image"].apply(extract_patient_id)
            patient_scores = df.groupby("patient").agg({
                "prob_PD": "mean"
            }).reset_index()

            patient_scores["final_prediction"] = np.where(
                patient_scores["prob_PD"] >= THRESHOLD, "PD", "Healthy"
            )

            overall_pred = patient_scores.iloc[0]["final_prediction"]
            avg_prob = patient_scores.iloc[0]["prob_PD"]

            # Final Result
            st.markdown("---")
            st.subheader("Diagnosis: ")
            st.markdown(f"### Final Prediction: **{overall_pred}**")

            # Download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 Download Prediction Report",
                csv,
                "PD_slice_predictions.csv",
                "text/csv"
            )

else:
    st.info("👆 Please upload MRI images to begin classification.")

st.markdown("---")
st.caption("Developed for Parkinson MRI Diagnosis - Powered by MobileNetV2")
