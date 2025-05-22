#!/bin/sh

image_name="myungjune/mineru-web-api-capstone"
tag="cpu"
cache_tag="build-cache"

docker buildx build \
  --platform linux/arm64,linux/amd64 \
  -t "${image_name}:${tag}" \
  --cache-from=type=registry,ref="${image_name}:${cache_tag}" \
  --cache-to=type=registry,ref="${image_name}:${cache_tag}",mode=max \
  --push .