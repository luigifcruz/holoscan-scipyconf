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


```
trtexec --onnx=cursednet.onnx --saveEngine=cursednet.engine
```