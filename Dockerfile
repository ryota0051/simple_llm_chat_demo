FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV POETRY_VERSION=1.7.0 \
    POETRY_HOME="/opt/poetry"

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    curl

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false

COPY poetry.lock pyproject.toml ./

RUN poetry install --no-dev
