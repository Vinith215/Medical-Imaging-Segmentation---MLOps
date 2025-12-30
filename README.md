# Medical-Imaging-Segmentation---MLOps

# 🏥 Medical Segmentation MLOps Pipeline

An end-to-end, production-grade MLOps framework for 3D medical image segmentation using **MONAI**, **Swin-UNETR**, and **AWS**.

---

## 📌 Project Overview
This repository provides a robust pipeline for converting raw DICOM images into 3D segmentation masks. It features automated preprocessing, transformer-based training, GPU-optimized inference, and real-time monitoring.

**Key Technologies:**
* **Framework:** MONAI, PyTorch 2.5
* **Architectures:** Swin-UNETR (Shifted Window Transformer), 3D U-Net
* **Infrastructure:** Docker, AWS (S3, ECR), GitHub Actions
* **MLOps:** MLflow (Tracking), Evidently AI (Monitoring), Streamlit (UI)

---

## 📂 Directory Structure
* `src/`: Core logic (Ingestion, Preprocessing, Model Factory, Inference).
* `config/`: Centralized parameters for models and transforms.
* `data/`: Versioned data lake (Raw, Processed, Metadata).
* `models/`: Registry for training checkpoints and production weights.
* `app.py`: Streamlit dashboard for clinical users.
* `main.py`: CLI Entry point for pipeline automation.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/med-segmentation-mlops.git](https://github.com/your-username/med-segmentation-mlops.git)
   cd med-segmentation-mlops

2. Environment Configuration: Create a .env file in the root directory:
    AWS_ACCESS_KEY_ID=your_key
    AWS_SECRET_ACCESS_KEY=your_secret
    S3_BUCKET_NAME=your-bucket-name
    MLFLOW_TRACKING_URI=http://localhost:5000

3. Install Dependencies:
    conda create -n med-mlops python=3.10
    conda activate med-mlops
    pip install -r requirements.txt


### 🚀 Pipeline Execution

Phase 1: Ingestion
Download data from S3 and convert DICOM folders to NIfTI volumes:
    python main.py ingest

Phase 2: Preprocessing
Resample volumes to 1.0mm isotropic spacing and normalize intensities:
    python main.py preprocess

Phase 3 & 4: Training
Start training the Swin-UNETR model with MLflow tracking:
    python main.py train

Phase 5: Inference
Run 3D sliding-window inference on a single scan:
    python main.py infer --input data/raw/sample.nii.gz --output outputs/mask.nii.gz


🐳 Deployment (Docker)

1. Build the Image:
    docker build -t med-seg-mlops .

2. Run the Dashboard:
    docker run -p 8501:8501 --gpus all med-seg-mlops

    Access the UI at http://localhost:8501.


📊 Monitoring & Drift Analysis

Generate an Evidently AI report to detect clinical data drift:
    python src/monitoring.py


🧪 Testing
Run the validation suite (requires pytest):
    pytest tests/


## Final Project Review

Your project is now fully architected and documented. Here is the final logic flow you have built:



1.  **Data enters via S3** and is logged into the `dataset_manifest.csv`.
2.  **Preprocessing** ensures clinical uniformity (Isotropic spacing).
3.  **Swin-UNETR** is trained, using **MLflow** to log every metric.
4.  **The Best Model** is promoted to `models/production/`.
5.  **GitHub Actions** packages the code into a **Docker** image.
6.  **Streamlit** provides the frontend for real-world clinical inference.
7.  **Evidently AI** watches for drift to ensure the model stays safe and accurate over time.

**Is there anything else you would like to refine in this pipeline, such as a specific CI.