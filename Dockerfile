FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN adduser jupyter

USER jupyter
