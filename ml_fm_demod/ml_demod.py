import sys
import queue
import time

try:
    import numpy as np
except ImportError:
    raise ImportError("This demo requires Numpy, but it could not be imported.")

try:
    import cupy as cp
except ImportError:
    raise ImportError("This demo requires cupy, but it could not be imported.")

try:
    from cupyx.scipy import signal as gpu
except ImportError:
    raise ImportError("This demo requires cupyx, but it could not be imported.")

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX
except ImportError:
    raise ImportError("This demo requires SoapySDR, but it could not be imported.")

try:
    import sounddevice as sd
except ImportError:
    raise ImportError("This demo requires SoundDevice, but it could not be imported.")

from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import InferenceOp
from holoscan.resources import UnboundedAllocator


sdr_fs = int(240e3)
audio_fs = int(48e3)
audio_buffer_size = int(48e3)
buffer_size = audio_buffer_size * (sdr_fs // audio_fs)
que = queue.Queue()
rx = None


class SignalGeneratorOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("rx_sig")

    def compute(self, op_input, op_output, context):
        buffer = np.zeros(buffer_size, dtype=cp.complex64)
        samples = 0

        while samples < buffer_size:
            res = sdr.readStream(rx, [buffer[samples:]], min(8192, buffer_size-samples), timeoutUs=int(1e18))
            samples += res.ret

        op_output.emit(cp.asarray(buffer.astype(cp.complex64)), "rx_sig")


class PreProcessorOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("rx_sig")
        spec.output("tensor")

    def compute(self, op_input, op_output, context):
        tensor = op_input.receive("rx_sig")
        tensor = gpu.resample(tensor, int(256e3)) # Model expects 256e3 samples.
        tensor = cp.stack([cp.real(tensor), cp.imag(tensor)])
        tensor = cp.ascontiguousarray(tensor)
        op_output.emit(dict(rx_sig=tensor), "tensor")


class PostProcessorOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("rx_sig")
        spec.output("rx_sig")

    def compute(self, op_input, op_output, context):
        sig = op_input.receive("rx_sig")['rx_sig']
        sig = cp.asarray(sig)[0, 0, :]
        sig = gpu.resample(sig, audio_buffer_size) # Model outputs only 32e3 samples.
        op_output.emit(sig, "rx_sig")


class DemodulateOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("rx_sig")
        spec.output("sig_out")

    def compute(self, op_input, op_output, context):
        sig = op_input.receive("rx_sig")
        x_angle = cp.unwrap(cp.angle(sig))
        sigd = cp.diff(x_angle)
        op_output.emit(sigd, "sig_out")


class ResampleOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("rx_sig")
        spec.output("sig_out")

    def compute(self, op_input, op_output, context):
        sig = op_input.receive("rx_sig")
        sig = gpu.resample_poly(sig, 1, sdr_fs // audio_fs, window="hamm").astype(cp.float32)
        op_output.emit(sig, "sig_out")


class SDRSinkOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("rx_sig")

    def compute(self, op_input, op_output, context):
        sig = op_input.receive("rx_sig")
        que.put_nowait(cp.asnumpy(sig))


class NeuralFmDemod(Application):
    def __init__(self):
        super().__init__()

    def compose(self):
        pool = UnboundedAllocator(self, name="allocator")
        src = SignalGeneratorOp(self, name="src")
        preprocessor = PreProcessorOp(self, name="preprocessor")
        inference = InferenceOp(self,
            allocator=pool,
            backend="trt",
            input_on_cuda=True,
            output_on_cuda=True,
            transmit_on_cuda=True,
            is_engine_path=True,
            pre_processor_map={
                "demod": ["rx_sig"],
            },
            model_path_map={
                "demod": "./cursednet.engine",
            },
            inference_map={
                "demod": ["rx_sig"],
            },
        )
        postprocessor = PostProcessorOp(self, name="postprocessor")
        sink = SDRSinkOp(self, name="sink")

        self.add_flow(src, preprocessor)
        self.add_flow(preprocessor, inference, {("tensor", "receivers")})
        self.add_flow(inference, postprocessor)
        self.add_flow(postprocessor, sink)


class StandardFmDemod(Application):
    def __init__(self):
        super().__init__()

    def compose(self):
        src = SignalGeneratorOp(self, name="src")
        demodulate = DemodulateOp(self, name="demodulate")
        resample = ResampleOp(self, name="resample")
        sink = SDRSinkOp(self, name="sink")

        self.add_flow(src, demodulate)
        self.add_flow(demodulate, resample)
        self.add_flow(resample, sink)


def audio_callback(outdata, *_):
    if not que.empty():
        outdata[:, 0] = que.get_nowait()
    else:
        outdata[:, 0] = 0.0


if __name__ == "__main__":
    print('#####################################')
    print('###     NEURAL FM DEMODULATOR     ###')
    print('#####################################')
    print('')
    print('Sample Holoscan pipeline showing real-time inference of IQ signals.')
    print('')
    print('Usage:')
    print('    Enable the neural-based demodulation with `--ml`.')
    print('    Optional: Choose the frequency of the FM radio with `-f [FREQ HZ]`.')
    print('    Optional: Select the sound device with `-d [INDEX]`.')
    print('')
    print("Available sound devices:")
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        print(f"  {idx}: {device['name']}")
    print('')

    use_ml = '--ml' in sys.argv

    device = None
    if '-d' in sys.argv:
        device = int(sys.argv[sys.argv.index('-d') + 1])

    fm_freq = 96500000
    if '-f' in sys.argv:
        fm_freq = int(sys.argv[sys.argv.index('-f') + 1])

    #
    # Setup Software Defined Radio.
    #

    try:
        args = dict(driver="rtlsdr")
    except ImportError:
        raise ImportError("Ensure SDR is connected and appropriate drivers are installed.")

    sdr = SoapySDR.Device(args)
    sdr.setSampleRate(SOAPY_SDR_RX, 0, sdr_fs)
    sdr.setFrequency(SOAPY_SDR_RX, 0, fm_freq)
    sdr.setGainMode(SOAPY_SDR_RX, 0, True)
    rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rx)

    #
    # Start output audio device.
    #

    stream = sd.OutputStream(blocksize=audio_buffer_size,
                             callback=audio_callback,
                             samplerate=audio_fs,
                             channels=1,
                             device=device)

    #
    # Start Holoscan pipeline.
    #

    if use_ml:
        app = NeuralFmDemod()
    else:
        app = StandardFmDemod()

    app.config("")
    stream.start()
    app.run()
