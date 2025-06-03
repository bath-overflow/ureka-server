## Setup

1. `pip install uv`
2. `uv sync`
3. `uv run pre-commit install`
4. `uv run uvicorn server.main:app --reload`


## Setup docker compsose

`docker-compose -f docker-compose.dev.yml up --build`

