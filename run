#!/bin/bash

docker pull tensorflow/serving

docker run -p 8501:8501 \
--name tfserving-universal-text-encoder \
--mount type=bind,source=/Users/youngguebae/Documents/projects/tfserving-universal-encoder/data/tfserving/nnlm-en-dim50,target=/models/universal_encoder \
-e MODEL_NAME=universal_encoder -t tensorflow/serving &
