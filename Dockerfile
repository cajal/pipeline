FROM ninai/pipeline:base
LABEL maintainer="Edgar Y. Walker, Fabian Sinz, Erick Cobos, Donnie Kim"

WORKDIR /data

# Install pipeline
COPY . /data/pipeline
RUN pip3 install -e pipeline/python/

ENTRYPOINT ["/bin/bash"]
