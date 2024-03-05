FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set bash as the default shell
ENV SHELL=/bin/bash
ENV http_proxy="http://proxy.san.gva.es:8080"
ENV https_proxy="http://proxy.san.gva.es:8080"
ENV ftp_proxy="http://proxy.san.gva.es:8080"
ENV no_proxy="127.0.0.1,localhost"
ENV HTTP_PROXY="http://proxy.san.gva.es:8080"
ENV HTTPS_PROXY="http://proxy.san.gva.es:8080"
ENV FTP_PROXY="http://proxy.san.gva.es:8080"

WORKDIR /app/

COPY dicom_drift_monitor.py /app/
COPY utils.py /app/

RUN mkdir /data/

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    apt-utils \
    vim \
    git

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# build with some basic python packages
RUN pip install \
    numpy==1.22.4 \
    highdicom==0.21.1 \
    monai==1.2.0 \
    pydicom==2.3.1 \
    SimpleITK==2.1.1.2 \
    typeguard==4.1.5 \
    scipy==1.11.2 \
    matplotlib==3.5.3 \
    scikit-learn==1.3.2 \
    umap-learn==0.5.5 \

CMD ["bash"]
