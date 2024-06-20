import logging
import socket
import numpy as np
from time import sleep
from scipy import signal

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.logger import LogLevel, set_log_level


class UdpRxOp(Operator):
    sock_fd: socket.socket = None
    data: bytearray

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("UdpRxOp")
        logging.basicConfig(level=logging.INFO)

    def initialize(self):
        Operator.initialize(self)
        self.logger.info("UDP RX Operator initialized")

        try:
            self.sock_fd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        except socket.error:
            self.logger.error("Failed to create socket")

        self.sock_fd.bind(("127.0.0.1", 8080))

    def setup(self, spec: OperatorSpec):
        spec.output("wave")

    def compute(self, op_input, op_output, context):
        while True:
            try:
                self.data = self.sock_fd.recvfrom(8000, socket.MSG_WAITALL)[0]
            except BlockingIOError:
                raise RuntimeError("socket connection broken")

            wave = np.frombuffer(self.data, dtype=np.complex64)
            op_output.emit(wave, "wave")
            return


class ResamplerOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("ResamplerOp")
        logging.basicConfig(level=logging.INFO)

    def initialize(self):
        Operator.initialize(self)
        self.logger.info("Resampler Operator initialized")

    def setup(self, spec: OperatorSpec):
        spec.input("wave")

    def compute(self, op_input, op_output, context):
        wave = op_input.receive("wave")
        assert isinstance(wave, np.ndarray)

        resampled_wave = signal.resample(wave, 100)

        print(wave.shape, resampled_wave.shape)


class SineUdpRx(Application):
    def compose(self):
        udp_rx_op = UdpRxOp(self, name="UdpRxOp")
        resampler_op = ResamplerOp(self, name="ResamplerOp")

        self.add_flow(udp_rx_op, resampler_op)


def main():
    app = SineUdpRx()
    app.run()


if __name__ == "__main__":
    main()