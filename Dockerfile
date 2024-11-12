FROM public.ecr.aws/ubuntu/ubuntu:20.04

RUN apt update

RUN apt install -y software-properties-common

RUN apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt install -y python3.10 python3-pip

RUN pip install ultralytics

RUN pip install roboflow

RUN pip install 'git+https://github.com/facebookresearch/segment-anything.git'

RUN pip install supervision

# Downdload sam weights
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# Try with a smaller model for testing.

# RUN curl -fsSL https://ollama.com/install.sh | sh

# RUN pip install ollama