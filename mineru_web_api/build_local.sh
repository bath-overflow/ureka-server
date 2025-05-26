#!/bin/bash

image_name="myungjune/mineru-web-api-capstone"
tag="cpu"
cache_tag="build-cache"

# Build for your local architecture only with --load
docker buildx build \
  --platform linux/$(uname -m | sed 's/x86_64/amd64/;s/arm64/arm64/') \
  -t "${image_name}:${tag}" \
  --cache-from=type=registry,ref="${image_name}:${cache_tag}" \
  --cache-to=type=registry,ref="${image_name}:${cache_tag}",mode=max \
  --load .