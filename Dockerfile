FROM tensorflow/tensorflow:latest-gpu-jupyter

# Install OpenCV package
RUN python3 -m pip install --no-cache-dir opencv-python-headless

RUN adduser jupyter
USER jupyter
