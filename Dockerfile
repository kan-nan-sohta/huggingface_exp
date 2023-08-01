ARG BASE_IMAGE=python:3.10.6-slim-bullseye
FROM ${BASE_IMAGE}

ARG PROJECT_NAME=huggingface-exp
ARG USER_NAME=kan_nan
ARG GROUP_NAME=kan_nan
ARG UID=1000
ARG GID=1000
ARG APPLICATION_DIRECTORY=/home/${USER_NAME}/${PROJECT_NAME}/working
ARG WORKING_PATH=working
ARG RUN_POETRY_INSTALL_AT_BUILD_TIME="false"

ENV DEBIAN_FRONTEND="noninteractive" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    PYTHONPATH=${APPLICATION_DIRECTORY}

RUN apt update
RUN apt install --no-install-recommends -y git curl make python3-pip libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-dev
RUN pip install poetry

# Add user. Without this, following process is executed as admin. 
RUN groupadd -g ${GID} ${GROUP_NAME} \
    && useradd -ms /bin/sh -u ${UID} -g ${GID} ${USER_NAME}

USER ${USER_NAME}
WORKDIR ${APPLICATION_DIRECTORY}

# If ${RUN_POETRY_INSTALL_AT_BUILD_TIME} = "true", install Python package by Poetry and move .venv under ${HOME}.
# This process is for CI (GitHub Actions). To prevent overwrite by volume of docker compose, .venv is moved under ${HOME}.
# COPY --chown=${UID}:${GID} ${WORKING_PATH}/pyproject.toml ${WORKING_PATH}/poetry.lock ${WORKING_PATH}/poetry.toml .
RUN test ${RUN_POETRY_INSTALL_AT_BUILD_TIME} = "true" && poetry install || echo "skip to run poetry install."
RUN test ${RUN_POETRY_INSTALL_AT_BUILD_TIME} = "true" && mv ${APPLICATION_DIRECTORY}/.venv ${HOME}/.venv || echo "skip to move .venv."
