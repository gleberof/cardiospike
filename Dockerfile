FROM python:3.8.6

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
     git build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev \
     libffi-dev curl libbz2-dev ca-certificates \
       && rm -rf /var/lib/apt/lists/* && apt-get clean

WORKDIR cardiospike

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

COPY pyproject.toml poetry.lock ./
RUN mkdir cardiospike && touch cardiospike/__init__.py
RUN poetry install
RUN rm -rf $HOME/.cache/pypoetry

COPY . .
