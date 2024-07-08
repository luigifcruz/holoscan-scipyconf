# SciPy Conference 2024 - Holoscan Demos

These are supporting demo projects using the [NVIDIA Holoscan SDK](https://developer.nvidia.com/holoscan-sdk) from our presentation at the SciPy Conference 2024.

### Requirements

- A Linux host running Docker and [NVIDIA Container Runtime](https://developer.nvidia.com/container-runtime).
- An NVIDIA Graphics Card.
- An output audio device with ALSA installed.
- A Software Defined Radio (for the neural FM demo). We recommend the RTL-SDR v3.

### Getting Started

Follow these steps to clone the repository and build a Docker image:

1. **Clone the Repository**

Open your terminal and run the following command to clone the repository to your computer:

```bash
$ git clone https://github.com/luigifcruz/holoscan-scipyconf
```
   
3. **Navigate to the Repository Directory**

Change to the repository directory:

```bash
$ cd holoscan-scipyconf
```

4. **Build the Docker Image**

Build the Docker image using the following command:

```bash
$ docker build -t holoscan-scipyconf .
```

6. **Run the Docker Container**

Start the Docker container with the necessary privileges and settings:

```bash
$ docker run --rm -it \
  --privileged \
  --gpus=all \
  --entrypoint bash \
  --device /dev/snd \
  -v .:/demos \
  holoscan-scipyconf
```

Privilege is necessary to access the Software Defined Radio via USB.

7. **Done!**

This setup will give you a Docker environment ready for running the NVIDIA Holoscan SDK demo projects. Follow the steps below for each example.

## Neural FM Demodulator

This demo receives an IQ stream from an FM broadcast radio station via a Software Defined Radio (SDR). The data is then processed by [Holoscan's Inference Operator](https://docs.nvidia.com/holoscan/sdk-user-guide/inference.html). The output can be directly played back using the audio interface. Additionally, this demo compares the neural network-based demodulation with the traditional demodulation algorithm.

https://github.com/luigifcruz/holoscan-scipyconf/assets/6627901/738a6949-3ec1-4b13-a13e-3e275e9f8def

1. **Navigate to the directory**

```bash
$ cd ml_fm_demod
```

2. **Compile TensorRT Engine**

It's necessary to manually compile the TensorRT Engine from the model's ONNX file:

```bash
$ trtexec --onnx=cursednet.onnx --saveEngine=cursednet.engine
```

3. **Run the traditional demodulation**

Before demodulating it with the ML model, let's hear how it should sound like by demodulating it via a traditional mathematical method.

To do this, run this command on the terminal. You might need to change the frequency (in Hertz) to a local radio broadcast station.

```bash
$ python ml_demod.py -f 96900000
```

The output sound card can also be selected via the `-d [INDEX]` command-line argument.

4. **Run the neural demodulation**

Now let's try to demodulate the same radio station using purely the machine learning model.

This can be done by passing the `--ml` argument to the launch command.

```bash
$ python ml_demod.py -f 96900000 --ml
```

The resulting audio should be a bit more noisy than the traditional demodulation.

## Basic Network Operator

https://github.com/luigifcruz/holoscan-scipyconf/assets/6627901/d12d7698-664a-47dd-847e-0b8f95abc617

This demo illustrates how to send and receive UDP packages using [Holoscan's Basic Network Operator](https://github.com/nvidia-holoscan/holohub/tree/main/operators/basic_network).

1. **Navigate to the directory**

```bash
$ cd basic_network_op
```

2. **Start UDP transmitter**

This pipeline is responsible for sending UDP packages to the UDP receiver. These are fixed-length packages containing sine and cosine waves.

```bash
$ python udp_tx.py
```

This transmitter can be replaced by the GNU Radio flowgraph included in the same directory. Add `--network=host` to the Docker container run to enable the UDP packages to be sent from the host.

3. **Start UDP receiver**

Receive the UDP packages sent from the transmitter and resample the waves using the CPU.

```bash
& python udp_rx.py
```

The GPU can be used for resampling instead of the CPU by simply passing the `--cuda` argument.

```bash
& python udp_rx.py --cuda
```
