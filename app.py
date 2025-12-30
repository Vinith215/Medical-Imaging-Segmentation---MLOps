import streamlit as st
import importlib
try:
    nib = importlib.import_module("nibabel")
except Exception:
    nib = None
import numpy as np
import os
import shutil
import uuid
from pathlib import Path

from config.global_config import GlobalConfig
from src.inference import SegmentationInference

# Page Configuration
st.set_page_config(page_title="Med-Seg AI Dashboard", layout="wide")

def save_uploaded_file(uploaded_file):
    """Saves file to temp_uploads with a unique session ID."""
    session_id = str(uuid.uuid4())
    temp_dir = GlobalConfig.DATA_DIR / "temp_uploads" / session_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path, temp_dir

def main():
    st.title("🏥 3D Medical Segmentation MLOps App")
    st.sidebar.header("Navigation")
    
    # 1. File Upload
    uploaded_file = st.file_uploader("Upload a 3D NIfTI Scan (.nii.gz)", type=["gz"])
    
    if uploaded_file:
        file_path, temp_dir = save_uploaded_file(uploaded_file)
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # 2. Trigger Inference
        if st.button("🚀 Run AI Segmentation"):
            with st.spinner("Processing 3D Volume with Swin-UNETR..."):
                engine = SegmentationInference()
                output_path = temp_dir / f"mask_{uploaded_file.name}"
                
                # Run the prediction
                prediction = engine.predict(str(file_path), str(output_path))
                
                # Store in session state for visualization
                if nib is None:
                    st.error("Required package 'nibabel' is not installed. Install it with: pip install nibabel")
                else:
                    st.session_state['original'] = nib.load(file_path).get_fdata()
                    st.session_state['mask'] = prediction
                    st.session_state['mask_path'] = output_path
                    st.success("Segmentation Complete!")

        # 3. Visualization Section
        if 'original' in st.session_state:
            st.divider()
            col1, col2 = st.columns([1, 1])
            
            vol = st.session_state['original']
            mask = st.session_state['mask']
            
            # Slice Slider
            max_slices = vol.shape[2]
            idx = st.slider("Select Slice (Z-Axis)", 0, max_slices-1, max_slices // 2)
            
            with col1:
                st.subheader("Original Scan")
                st.image(vol[:, :, idx], use_container_width=True, clamp=True)
                
            with col2:
                st.subheader("AI Segmentation Overlay")
                # Simple overlay: showing mask where value > 0
                overlay = np.stack([vol[:, :, idx]] * 3, axis=-1)
                overlay[mask[:, :, idx] > 0] = [1, 0, 0] # Color red
                st.image(overlay, use_container_width=True)

            # 4. Download Result
            with open(st.session_state['mask_path'], "rb") as f:
                st.download_button(
                    label="📥 Download Segmentation Mask",
                    data=f,
                    file_name=f"ai_mask_{uploaded_file.name}",
                    mime="application/gzip"
                )

if __name__ == "__main__":
    main()