import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import tempfile
import os

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Garbage Detector",
    page_icon="🗑️",
    layout="wide"
)

# --------------------------------------------------
# Load model
# --------------------------------------------------

MODEL_PATH = "runs/detect/train-5/weights/best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🗑️ AI-Based Garbage Detection")
st.write(
    "Upload an image and the YOLO object detection model "
    "will identify garbage objects present in the image."
)

# --------------------------------------------------
# System information
# --------------------------------------------------

with st.sidebar:
    st.header("Model Information")

    st.write("**Model:** YOLO26n")
    st.write("**Classes:** 42")
    st.write("**Training Epochs:** 6")
    st.write("**Image Size:** 320 × 320")

    if torch.xpu.is_available():
        st.success("Intel Arc XPU Available")
        device = "xpu"
    else:
        st.warning("XPU unavailable — using CPU")
        device = "cpu"

# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a garbage image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Input Image")

    st.image(
        image,
        caption="Uploaded Image",
        width=500
    )

    # --------------------------------------------------
    # Detection button
    # --------------------------------------------------

    if st.button("🔍 Detect Garbage", type="primary"):

        with st.spinner("Detecting garbage..."):

            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as temp_file:

                image.save(temp_file.name)
                temp_path = temp_file.name

            # Run YOLO
            results = model.predict(
                source=temp_path,
                imgsz=320,
                conf=0.10,
                device=device,
                verbose=False
            )

            result = results[0]

            # Generate annotated image
            annotated_image = result.plot()
            annotated_image = annotated_image[:, :, ::-1]

            # --------------------------------------------------
            # Display result
            # --------------------------------------------------

            st.subheader("Detection Result")

            st.image(
                annotated_image,
                caption="YOLO Detection",
                width=500
            )

            # --------------------------------------------------
            # Detection information
            # --------------------------------------------------

            if len(result.boxes) > 0:

                st.subheader("Detected Objects")

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    class_name = model.names[class_id]

                    st.write(
                        f"**{class_name}** — "
                        f"{confidence:.2%} confidence"
                    )

            else:

                st.warning(
                    "No garbage objects were detected in this image."
                )

            # Remove temporary file
            os.remove(temp_path)