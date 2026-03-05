ARG INTERFACES_IMAGE=gigas/interfaces:dev
ARG ROS_DISTRO=humble
FROM ${INTERFACES_IMAGE}

ARG ROS_DISTRO=humble

SHELL ["/bin/bash", "-lc"]

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/workspace/.ultralytics

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    libgl1 libglib2.0-0 \
    libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

# Python toolchain
# Pin setuptools < 80. Setuptools >= 80 can break colcon --symlink-install for ament_python
# packages with: "error: option --editable not recognized".
RUN python3 -m pip install --no-cache-dir --upgrade \
    "pip<25" \
    "setuptools<80" \
    wheel

# Enforce Python 3.10 (ROS2 Humble base uses Python 3.10)
RUN python3 -c "import sys; assert sys.version_info[:2] == (3, 10), f'Expected Python 3.10, got {sys.version}'"

# Torch (Detectron2 pin set, CUDA 11.3 wheels)
# torchvision is required by stage3_ripeness.py (mobilenet + transforms)
RUN python3 -m pip install --no-cache-dir \
    "torch==1.12.1+cu113" \
    "torchvision==0.13.1+cu113" \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Python deps used by this pipeline
RUN python3 -m pip install --no-cache-dir \
    "numpy==1.26.4" \
    "scipy==1.15.3" \
    "opencv-python==4.11.0.86" \
    "pyrealsense2==2.56.5.9235" \
    "ultralytics==8.3.252"

# Copy pipeline code
COPY tomato_pipeline /workspace/tomato_pipeline

# Fanuc interface python helpers (for ros2_messages)
RUN mkdir -p /opt/fanuc_interface/src
COPY GIGAS-CODE/ros2-fanuc-robot-interface/workspaces/fanuc_interface/src/fanuc_interface /opt/fanuc_interface/src/fanuc_interface

WORKDIR /workspace

# Build tomato_pipeline as a ROS2 ament_python package
RUN mkdir -p /workspace/src && \
    ln -s /workspace/tomato_pipeline /workspace/src/tomato_pipeline && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    source /opt/robot_interfaces/setup.bash && \
    colcon build --symlink-install --base-paths /workspace/src

# Dev convenience
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc
RUN echo "source /workspace/install/setup.bash" >> /root/.bashrc

# Ensure Fanuc python helpers are importable for both source-root and package-root layouts.
ENV PYTHONPATH=/opt/fanuc_interface/src:/opt/fanuc_interface/src/fanuc_interface

# Source workspace on container entrypoint
RUN sed --in-place --expression \
    '$isource "/workspace/install/setup.bash"' \
    /ros_entrypoint.sh
