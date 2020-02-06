#!/bin/bash

DOCKER_IMAGE=$1
CONTAINER_NAME=$2
DOCKER_PULL=$3

docker stop "$CONTAINER_NAME"
docker rm "$CONTAINER_NAME"

if [ "$DOCKER_PULL" = "--docker_pull" ]
then
  docker pull tensorflow/serving:nightly
fi

docker run --rm -p 8501:8501 \
--name "$CONTAINER_NAME" \
--mount type=bind,source=/Users/youngguebae/Documents/projects/tfserving-universal-encoder/servables/universal-sentence-encoder-multilingual,target=/models/universal-sentence-encoder-multilingual \
-e MODEL_NAME=universal-sentence-encoder-multilingual -t "$DOCKER_IMAGE" &
