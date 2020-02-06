
## Export model

Installs the required packages via pipenv.
```
$ pipenv install --skip-lock
```
Activates this project's virtualenv.
```
$ pipenv shell
```

Exports the model from the tfhub.
```
$ python export_model.py --module_url TFHUB_MODULE_URL --export_path YOUR_PATH_TO_EXPORT
```
For example:
```
$ python export_model.py --module_url https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1 --export_path servables/nnlm-en-dim50/1
```
```
$ python export_model.py --module_url https://tfhub.dev/google/universal-sentence-encoder-multilingual/3 --export_path servables/universal-sentence-encoder-multilingual/3
```

## Run Tensorflow Serving
This has a `tensorflow_text` support issue:
* https://github.com/tensorflow/serving/issues/1490
* https://github.com/google/sentencepiece/issues/325
```
./run.sh tensorflow/serving tfserving-universal-encoder --docker_pull
```
Therefore, build Tensorflow Serving manually for a temporary workaround as below.

## Build Tensorflow Serving manually
When you got stuck in `[3,154 / 4,665] @org_tensorflow//tensorflow/core/kernels:matmul_op`,
You are likely running out of memory.
Try reducing number of parallel builds by passing '--jobs 1 --local_resources 2048,.5,1.0',
which would instruct bazel to spawn no more than one compiler process at the time.

- https://github.com/tensorflow/tensorflow/issues/349
- https://github.com/tensorflow/serving/issues/1251
- https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/building_with_docker.md

Increase your CPU, memory and available disk space of Docker as well!
```
$ docker build -t tensorflow/serving:with-latest-tensorflow-text --build-arg TF_SERVING_BUILD_OPTIONS="--local_resources 2048,0.5,1.0" .
```

## Run docker container with custom build
```
./run.sh tensorflow/serving:with-latest-tensorflow-text tfserving-universal-encoder
```

Enter the docker container with shell command.
```
$ docker exec -it YOUR_DOCKER_CONTAINER_NAME /bin/bash
```

```
$ docker exec -it tfserving-universal-encoder /bin/bash
```

