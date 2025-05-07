FROM python:3.12-slim

# Set the working directory
WORKDIR /app

COPY pyproject.toml uv.lock pytest.ini ./

RUN pip install --no-cache-dir uv==0.6.3 && \
    uv sync --no-dev

COPY ./server ./server

ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "--no-dev" , "uvicorn", "--host", "0.0.0.0", "--port", "8000", "server.main:app"]

