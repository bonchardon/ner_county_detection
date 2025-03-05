FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip \
    && pip3 install torch transformers accelerate

WORKDIR /app

COPY . /app

CMD ["python3", "main.py"]
