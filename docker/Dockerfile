FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        wget \
        curl \
        git \
        vim \
        python3-pip \
        lsb-release \
        gnupg 


COPY ./. /root/frustum-convnet/
RUN chmod -R +x /root/frustum-convnet/*

# # # Link the default shell 'sh' to Bash.
# RUN ln -s /bin/bash /bin/sh 

# # Force color prompt in terminal.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

# CUDA 11.7
RUN pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

# boost cpp
RUN apt update && \
    apt install -y libboost-dev

# Add the following line to get native library of OpenCV.
RUN apt-get update && apt-get install -y libopencv-dev 

# opencv
RUN pip3 install opencv-python
# RUN apt-get update && \
#     apt-get install -y libgl1-mesa-dev

# matplotlib
RUN pip3 install matplotlib

# scipy
RUN pip3 install scipy

# yaml
RUN pip3 install PyYAML

# # tensorflow but only for tensorboard
# RUN pip3 install tensorflow===2.8.2 

RUN apt-get update && apt-get install gnuplot -y
RUN apt-get update && apt-get install -y evince poppler-data

#laspy
RUN python3 -m pip install "laspy[lazrs,laszip]"

#mmdetection
RUN pip3 install --no-cache-dir openmim==0.3.0
RUN pip3 install mmcv-full==1.4.0 mmdet==2.20.0
RUN git clone https://github.com/open-mmlab/mmdetection3d.git
RUN cd mmdetection3d &&  pip3 install -e .

# torchpack
RUN pip3 install torchpack

# mpi4py
RUN apt update &&\
    apt-get install libopenmpi-dev -y &&\
    pip3 install mpi4py

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

ENTRYPOINT []

CMD ["/bin/bash"]
