## Setup

1. `pip install uv`
2. `uv sync`
3. `uv run pre-commit install`
4. `uv run uvicorn main:app`


## Setup docker compsose

`docker-compose -f docker-compose.dev.yml up --build`

## To run unstructured-related tests

1. In VS Code, open `devcontainer/devcontainer.json`
2. Command Palette -> Dev Containers: Rebuild and Reopen in Container
3. Try running the ipynb files