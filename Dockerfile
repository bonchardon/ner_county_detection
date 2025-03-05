FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies (python3-pip, python3-dev, git, etc.)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip3 install --upgrade pip

# Install PyTorch for CUDA 11.3
RUN pip3 install torch==1.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install Transformers and Accelerate
RUN pip3 install transformers accelerate

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Default command to run your model script
CMD ["python3", "main.py"]
