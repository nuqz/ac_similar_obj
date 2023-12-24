FROM tensorflow/tensorflow:latest-gpu-jupyter

# Install additional packages
RUN python3 -m pip install --no-cache-dir -U \
    tensorboard-plugin-profile \
    tensorflow-probability \
    opencv-python-headless

RUN adduser jupyter
USER jupyter
