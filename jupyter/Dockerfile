FROM ninai/pipeline:latest

MAINTAINER Fabian Sinz <sinz@bcm.edu>

RUN pip3 install jupyter && \
    cp /data/pipeline/jupyter/jupyter_notebook_config.py /root/.jupyter/
EXPOSE 8888

WORKDIR /notebooks

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["jupyter notebook"]
