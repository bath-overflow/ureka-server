# MinerU 기반 PDF -> MD Web API

## Images built
- myungjune/mineru-web-api-capstone:latest : GPU에서 실행하는 이미지
    - `magic-pdf.json`에서 `"device-mode":"cuda"`
- myungjune/mineru-web-api-capstone:cpu : CPU에서 실행하는 이미지
    - `magic-pdf.json`에서 `"device-mode":"cpu"`

## Built by

```
source build_script.sh
```

## Start command

- GPU

```bash
docker run --rm -it --gpus=all -p 8888:8888 mineru-api
```

- CPU

```bash
docker run --rm -it -p 8888:8888 mineru-api
```

- docker-compose.ev.yml 이용하는 경우

README.md에서 설명하는대로 `docker-compose up` 커맨드 사용

## Usage

Refer to http://127.0.0.1:8888/docs