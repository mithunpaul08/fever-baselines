FROM continuumio/miniconda3

ENTRYPOINT ["/bin/bash"]

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN mkdir /fever/
RUN mkdir /fever/src
RUN mkdir /fever/config
RUN mkdir /fever/scripts
RUN mkdir /fever/data/
RUN mkdir /fever/data/fever/
RUN mkdir /fever/data/fever-data/
RUN mkdir /fever/data/models/
RUN mkdir /fever/data/fever-data-ann/


VOLUME /fever/

ADD requirements.txt /fever/
ADD *.yml /fever/
ADD src /fever/src/
ADD config /fever/config/
ADD scripts /fever/scripts/
ADD data/models/  /fever/data/models/
ADD data/fever-data/ /fever/data/fever-data/
#ADD data/fever/ /fever/data/fever/
ADD data/fever-data-ann/ fever/data/fever-data-ann/

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    zip \
    gzip \
    make \
    automake \
    gcc \
    build-essential \
    g++ \
    cpp \
    libc6-dev \
    man-db \
    autoconf \
    pkg-config \
    unzip

RUN conda update -q conda
RUN conda info -a
RUN conda create -q -n fever python=3.6
#RUN conda env create -f environment.yml

WORKDIR /fever/

RUN . activate fever
#RUN source activate hw4mithun
RUN conda install -y pytorch=0.4.0=py36_cuda8.0.61_cudnn7.1.2_1 torchvision=0.2.1=py36_1 -c pytorch
#pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
#pip3 install torchvision
RUN pip install -r requirements.txt
#RUN python src/scripts/prepare_nltk.
RUN PYTHONPATH=src
RUN python src/scripts/rte/da/eval_da.py data/fever/fever.db  data/fever/dev.ns.pages.p1.jsonl --param_path config/fever_nn_ora_sent_hw4.json --randomseed 1234 --slice 100
