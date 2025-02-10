# Use the official Ubuntu image
FROM python:3.12-bullseye

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install essential tools
RUN apt update && apt install -y \
    build-essential \
    git \
    && apt clean

# To give openCV dependencies, necessary for the ocr
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install other dependencies
RUN apt install -y \
    poppler-utils \
    tesseract-ocr 

# Set working directory
WORKDIR /workspaces

# Install Python dependencies inside a virtual environment
COPY requirements.txt /workspaces/requirements.txt
RUN python -m venv /workspaces/.venv
RUN /workspaces/.venv/bin/pip install --upgrade pip
RUN /workspaces/.venv/bin/pip install --no-cache-dir -r /workspaces/requirements.txt