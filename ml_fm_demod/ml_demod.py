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

# Demodulation and Radio Settings
fm_freq = 435000000
sdr_fs = int(256e3)
audio_fs = int(32e3)
audio_buffer_size = int(32e3)
buffer_size = audio_buffer_size * (sdr_fs // audio_fs)

try:
    args = dict(driver="rtlsdr")
except ImportError:
    raise ImportError("Ensure SDR is connected and appropriate drivers are installed.")

# SoapySDR Config
sdr = SoapySDR.Device(args)
sdr.setSampleRate(SOAPY_SDR_RX, 0, sdr_fs)
sdr.setFrequency(SOAPY_SDR_RX, 0, fm_freq)
rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

# Start streams and allocate buffers
buffer = np.zeros(buffer_size, dtype=cp.complex64)
sdr.activateStream(rx)
que = queue.Queue()


# Audio Config 

def audio_callback(outdata, *_):
    if not que.empty():
        outdata[:, 0] = que.get_nowait()
    else:
        outdata[:, 0] = 0.0


stream = sd.OutputStream(blocksize=audio_buffer_size,
                         callback=audio_callback,
                         samplerate=audio_fs,
                         channels=1)


class SignalGeneratorOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("rx_sig")

    def compute(self, op_input, op_output, context):
        samples = 0
        while samples < buffer_size:
            res = sdr.readStream(rx, [buffer[samples:]], min(8192, buffer_size-samples), timeoutUs=int(1e18))
            samples += res.ret
        a = cp.asarray(buffer.astype(cp.complex64))
        op_output.emit(a, "rx_sig")


class PreProcessorOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("rx_sig")
        spec.output("tensor")

    def compute(self, op_input, op_output, context):
        sig = op_input.receive("rx_sig")
        tensor = cp.zeros((1, 2, buffer_size), dtype=cp.float32)
        tensor[0, 0, :] = cp.real(sig)
        tensor[0, 1, :] = cp.imag(sig)
        op_output.emit(tensor, "tensor")
        


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
        op_output.emit(
            gpu.resample_poly(sig, 1, sdr_fs // audio_fs, window="hamm").astype(cp.float32),
            "sig_out",
        )


class SDRSinkOp(Operator):
    def __init__(self, *args, shape=(512, 512), **kwargs):
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
                "demod": "./model_trt.engine",
            },
            inference_map={
                "demod": ["rx_sig"],
            },
        )
        sink = SDRSinkOp(self, name="sink")

        self.add_flow(src, preprocessor)
        self.add_flow(preprocessor, inference, {("tensor", "receivers")})
        self.add_flow(inference, sink)


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


if __name__ == "__main__":
    use_ml = '--ml' in sys.argv

    if use_ml:
        app = NeuralFmDemod()
    else:
        app = StandardFmDemod()

    app.config("")
    stream.start()
    app.run()