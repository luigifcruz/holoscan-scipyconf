import logging
import socket
import numpy as np
from time import sleep

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.logger import LogLevel, set_log_level


class SineGenOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("SineGenOp")
        logging.basicConfig(level=logging.INFO)

    def initialize(self):
        Operator.initialize(self)
        self.logger.info("Sine Generator Operator initialized")

    def setup(self, spec: OperatorSpec):
        spec.output("wave")

    def compute(self, op_input, op_output, context):
        sine_wave = np.sin(np.linspace(0, 2 * np.pi, 1000, dtype=np.float32)) + \
                    np.cos(np.linspace(0, 2 * np.pi, 1000, dtype=np.float32)) * 1j

        op_output.emit(sine_wave, "wave")


class UdpTxOp(Operator):
    sock_fd: socket.socket = None

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("UdpTxOp")
        logging.basicConfig(level=logging.INFO)

    def initialize(self):
        Operator.initialize(self)
        self.logger.info("UDP TX Operator initialized")

        try:
            self.sock_fd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        except socket.error:
            self.logger.error("Failed to create socket")

    def setup(self, spec: OperatorSpec):
        spec.input("wave")

    def compute(self, op_input, op_output, context):
        wave = op_input.receive("wave")
        assert isinstance(wave, np.ndarray)

        if self.sock_fd.sendto(wave, ("127.0.0.1", 8080)) == 0:
            raise RuntimeError("socket connection broken")


class SineUdpTx(Application):
    def compose(self):
        sine_gen_op = SineGenOp(self, name="SineGenOp")
        udp_tx_op = UdpTxOp(self, name="UdpTxOp")

        self.add_flow(sine_gen_op, udp_tx_op)


def main():
    app = SineUdpTx()
    app.run()


if __name__ == "__main__":
    main()