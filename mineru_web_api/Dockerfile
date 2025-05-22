FROM python:3.12-slim-bookworm AS base

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1


FROM base AS build

# Update the package list and install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download models
RUN --mount=type=bind,source=download_models.py,target=download_models.py \
    python -m venv /app/venv && \
    . /app/venv/bin/activate && \
    pip install huggingface_hub && \
    ./download_models.py

# Build Python dependencies
RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    . /app/venv/bin/activate && \
    pip install -r requirements.txt

FROM base AS prod

# Copy Python dependencies and models from the build stage
COPY --from=build /app/venv /app/venv
COPY --from=build /opt/models /opt/models
COPY --from=build /opt/layoutreader /opt/layoutreader

# Update the package list and install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the app and its configuration file
COPY entrypoint.sh /app/entrypoint.sh
COPY magic-pdf.json /root/magic-pdf.json
COPY app/ /app/

# Expose the port that FastAPI will run on
EXPOSE 8888

# Command to run FastAPI using Uvicorn, pointing to app.py and binding to 0.0.0.0:8888
ENTRYPOINT [ "/app/entrypoint.sh" ]
CMD ["--host", "0.0.0.0", "--port", "8888"]
