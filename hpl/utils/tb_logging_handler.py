from logging import StreamHandler

from pytorch_lightning.loggers import TensorBoardLogger

TB_LOGGER = None


def set_tb_logger(tb_logger: TensorBoardLogger):
    global TB_LOGGER
    TB_LOGGER = tb_logger


class TBLoggingHandler(StreamHandler):
    def __init__(self, logger: TensorBoardLogger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = logger.experiment

    def emit(self, record):
        msg = self.format(record)
        self.writer.add_text("log", msg)
