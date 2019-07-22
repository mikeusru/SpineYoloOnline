FROM nvidia/cuda:9.0-base

# adapted from https://github.com/ceshine/Dockerfiles/blob/master/cuda/fastai/Dockerfile

ARG PYTHON_VERSION=3.6
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda


# Instal basic utilities
RUN apt-get update && \
  apt-get install -y --no-install-recommends git wget ffmpeg unzip bzip2 sudo build-essential && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
  wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
  echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
  /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
  rm -rf /tmp/* && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN conda install -y --quiet python=$PYTHON_VERSION

RUN  pip install --upgrade pip

COPY requirements/dev.txt /requirements.txt

RUN pip install -r /requirements.txt

COPY . /app
WORKDIR /app

CMD ["bash"]
