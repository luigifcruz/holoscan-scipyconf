import sys
import time
import logging
import socket
from time import sleep

import numpy as np
import cupy as cp

from scipy import signal as cpu 
from cupyx.scipy import signal as gpu

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.logger import LogLevel, set_log_level


class UdpRxOp(Operator):
    sock_fd: socket.socket = None
    packet_size: int
    data: bytearray

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("UdpRxOp")
        logging.basicConfig(level=logging.INFO)

        self.packet_size = kwargs['packet_size'] * 8
        # Check if the packet size fits in a UDP packet.
        assert self.packet_size <= 8000

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
                self.data = self.sock_fd.recvfrom(self.packet_size, socket.MSG_WAITALL)[0]
            except BlockingIOError:
                raise RuntimeError("socket connection broken")

            wave = np.frombuffer(self.data, dtype=np.complex64)
            op_output.emit(wave, "wave")
            return


class GatherOp(Operator):

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("GatherOp")
        logging.basicConfig(level=logging.INFO)

        self.batches = kwargs['batches']
        self.input_shape = kwargs['input_shape']

    def initialize(self):
        Operator.initialize(self)
        self.logger.info("Gather Operator initialized")

        self.iterator = 0
        self.data = np.zeros((self.batches, *self.input_shape), dtype=np.complex64) 

    def setup(self, spec: OperatorSpec):
        spec.input("data_in")
        spec.output("data_out")

    def compute(self, op_input, op_output, context):
        data_in = op_input.receive("data_in")

        self.data[self.iterator, :] = data_in
 
        if self.iterator < self.batches - 1:
            self.iterator += 1
        else:
            op_output.emit(self.data, "data_out")
            self.iterator = 0



class ResamplerOp(Operator):

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("ResamplerOp")
        logging.basicConfig(level=logging.INFO)

        self.rate = kwargs['rate']
        self.cuda = kwargs['cuda']

    def initialize(self):
        Operator.initialize(self)
        self.logger.info("Resampler Operator initialized")

    def setup(self, spec: OperatorSpec):
        spec.input("wave")

    def compute(self, op_input, op_output, context):
        wave = op_input.receive("wave")

        st = time.time()
        if self.cuda:
            resampled_wave = gpu.resample(wave, self.rate, axis=1)
        else:
            resampled_wave = cpu.resample(wave, self.rate, axis=1)
        et = time.time()

        print(wave.shape, resampled_wave.shape, f"Processing time: {(et - st) * 100} ms")


class SineUdpRx(Application):
    def compose(self):
        batches = 8192
        number_of_elements = 1000
        resample_rate = 10
        use_cuda = '--cuda' in sys.argv

        if use_cuda:
            print("Using CUDA for resampling.")

        udp_rx_op = UdpRxOp(self, name="UdpRxOp", packet_size=number_of_elements)
        gather_op = GatherOp(self, name="GatherOp", batches=batches, input_shape=(number_of_elements,))
        resampler_op = ResamplerOp(self, name="ResamplerOp", rate=(number_of_elements // resample_rate), cuda=use_cuda)

        self.add_flow(udp_rx_op, gather_op)
        self.add_flow(gather_op, resampler_op)


def main():
    app = SineUdpRx()
    app.run()


if __name__ == "__main__":
    main()
