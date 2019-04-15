#FROM datajoint/datajoint:latest
FROM ninai/pipeline:TF-GPU
LABEL maintainer="Edgar Y. Walker, Fabian Sinz, Erick Cobos, Donnie Kim"

WORKDIR /data

# Install commons
RUN git clone https://github.com/atlab/commons.git && \
    pip3 install commons/python && \
    rm -r commons

# Uninstall dlc and then Install my version of deeplabcut
RUN pip3 uninstall -y deeplabcut
RUN pip3 install git+https://github.com/DonnieKim411/DeepLabCut.git

# Install pipeline
COPY . /data/pipeline
RUN pip3 install -e pipeline/python/

ENTRYPOINT ["/bin/bash"]
