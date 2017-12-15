#FROM datajoint/datajoint:latest
FROM ninai/pipeline:base
LABEL maintainer="Edgar Y. Walker, Fabian Sinz, Erick Cobos"

WORKDIR /data

# --- install scanreader
RUN \
  git clone https://github.com/atlab/scanreader.git && \
  pip3 install -e scanreader

## --- install pipeline
COPY . /data/pipeline
RUN \
  pip3 install -e pipeline/python/

RUN git clone https://github.com/atlab/commons.git && \
    pip3 install -e commons/python

ENTRYPOINT ["python3", "/data/pipeline/python/scripts/populate-minion.py"]
