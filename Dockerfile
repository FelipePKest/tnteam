# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    SC2PATH=/workspace/tnteam/3rdparty/StarCraftII \
    PYTHONPATH=/workspace/tnteam/src \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git python3.10 python3.10-dev python3-pip python3-setuptools \
        python3-wheel libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
        libxi6 libfontconfig1 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /workspace/tnteam

COPY docker/requirements.txt /tmp/tnteam-requirements.txt
RUN python -m pip install --upgrade "pip<25" wheel setuptools && \
    python -m pip install \
        torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
        --extra-index-url https://download.pytorch.org/whl/cu117 && \
    python -m pip install torch-scatter \
        -f https://data.pyg.org/whl/torch-1.13.1+cu117.html && \
    python -m pip install -r /tmp/tnteam-requirements.txt && \
    python -c "import torch; print(torch.__version__)"

COPY . /workspace/tnteam

RUN mkdir -p /workspace/tnteam/naht_results /workspace/tnteam/3sv5z \
        /workspace/tnteam/uncntrl_agents /tmp/matplotlib && \
    python -m py_compile train_marie_naht_3sv5z.py train_poam_marie_utd_3sv5z.py && \
    python -c "from modules.marie import MARIEPolicy; print('tnteam MARIE import OK')"

CMD ["bash"]
