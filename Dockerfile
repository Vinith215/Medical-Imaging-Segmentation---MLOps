# Use a lightweight but GPU-optimized PyTorch base
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies for SimpleITK and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the project structure
COPY . .

# Ensure data and models directories exist
RUN mkdir -p data/raw data/processed data/metadata models/checkpoints models/production outputs

# Expose port for Streamlit
EXPOSE 8501

# Default command: Runs the Streamlit dashboard
# You can override this to run training or ingestion via docker run
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]