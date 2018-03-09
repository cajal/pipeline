#FROM datajoint/datajoint:latest
FROM ninai/pipeline:base
LABEL maintainer="Edgar Y. Walker, Fabian Sinz, Erick Cobos"

WORKDIR /data

# Install commons
RUN git clone https://github.com/atlab/commons.git && \
    pip3 install commons/python && \
    rm -r commons

# Install pipeline
COPY . /data/pipeline
RUN pip3 install -e pipeline/python/

ENTRYPOINT ["/bin/bash"]
