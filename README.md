# SciPy Conference 2024 - Holoscan Demos

```
$ git clone https://github.com/luigifcruz/holoscan-scipyconf
$ cd holoscan-scipyconf
$ docker build -t holoscan-scipyconf .
$ docker run --rm -it \
  --privileged \
  --gpus=all \
  --entrypoint bash \
  --device /dev/snd \
  -v .:/demos \
  holoscan-scipyconf
```

## Neural FM Demodulator

https://github.com/luigifcruz/holoscan-scipyconf/assets/6627901/738a6949-3ec1-4b13-a13e-3e275e9f8def

```
trtexec --onnx=cursednet.onnx --saveEngine=cursednet.engine
```

## Basic Network Operator

https://github.com/luigifcruz/holoscan-scipyconf/assets/6627901/d12d7698-664a-47dd-847e-0b8f95abc617



