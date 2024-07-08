FROM nvcr.io/nvidia/clara-holoscan/holoscan:v1.0.3-dgpu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update 

RUN apt install -y libsoapysdr0.8 python3-soapysdr soapysdr-tools
RUN apt install -y libportaudio2

RUN python -m pip install sounddevice astropy
RUN python -m pip install https://github.com/cupy/cupy/releases/download/v13.2.0/cupy_cuda12x-13.2.0-cp310-cp310-manylinux2014_x86_64.whl

WORKDIR /demos