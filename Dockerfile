FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-kor \
    libgl1 \
    # Clean up to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory
WORKDIR /app

COPY pyproject.toml uv.lock pytest.ini ./

# Copy over .env file
COPY .env ./

RUN pip install --no-cache-dir uv==0.6.3 && \
    uv sync --no-dev

# Install pip
RUN uv run python -m ensurepip --upgrade

COPY ./server ./server

ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "--no-dev" , "uvicorn", "--host", "0.0.0.0", "--port", "8000", "server.main:app"]

