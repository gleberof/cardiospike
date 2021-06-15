FROM nvidia/cuda:10.2-base
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y python3.8 python3-pip \
    python3.8-dev build-essential libgeos-dev liblzma-dev libssl-dev libbz2-dev curl python-dev git libffi-dev \
       && rm -rf /var/lib/apt/lists/*


# install pyenv
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> .bashrc
RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> .bashrc
RUN echo 'eval "$(pyenv init -)"' >> .bashrc

# install specific python version
RUN pyenv install 3.8.6 && pyenv global 3.8.6 && pyenv rehash

# install poetry and our package
ENV POETRY_NO_INTERACTION=1 \
    # send python output directory to stdout
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# add poetry and stuff to path
ENV POETRY_HOME="$HOME/opt/poetry" \
    VENV_PATH="$HOME/.local/"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH" POETRY_VERSION=1.1.6

RUN mkdir $HOME/opt/ && \
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3.8 - &&\
    poetry config virtualenvs.create false

WORKDIR cardiospike

COPY pyproject.toml poetry.lock ./
RUN poetry install
COPY . .
RUN poetry install
RUN rm -rf $HOME/.cache/pypoetry
