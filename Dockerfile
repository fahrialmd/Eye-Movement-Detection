# Use the NVIDIA TensorRT image as the base
FROM nvcr.io/nvidia/tensorrt:22.10-py3

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-22-dev \
    libv4l-dev \
    v4l-utils \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libgl1-mesa-glx \
    gfortran \
    python3-dev \
    python3-pip \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /workspace

# Copy all files from the local 'tensorrt-container' directory to the container's '/workspace' directory
COPY . /workspace

# # Install requirements
RUN pip install opencv-python \
    && pip install serial \ 
    && pip install torch==2.0.0 \ 
    && pip install torchvision==0.15.1 

# Set the default command to open a bash shell
CMD ["python", "deploy.py"]
