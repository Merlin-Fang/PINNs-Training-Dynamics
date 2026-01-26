import os
import logging
from tabulate import tabulate
from typing import Dict

class Logger:
    def __init__(
        self,
        name: str = 'main',
        handler_type: str = 'stream',
        log_info: Dict = None,
    ):
        self.logger = logging.getLogger(name)
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            '[%(asctime)s - %(name)s - %(levelname)s] %(message)s',
            datefmt="%H:%M:%S"
        )

        if handler_type == 'stream':
            handler = logging.StreamHandler()
        elif handler_type == 'file':
            log_dir = log_info.get('log_dir', './logs')
            file_name = log_info.get('file_name', f'{name}.log')
            os.makedirs(log_dir, exist_ok=True)
            handler = logging.FileHandler(
                os.path.join(log_dir, file_name)
            )
        else:
            raise ValueError(f"Unknown handler_type: {handler_type}")

        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def info(self, message):
        self.logger.info(message)

    def record(self, step, log_dict: Dict, start_time, end_time):
        # log_items = [
        #     [key, f"{log_dict[key]:.6f}"]
        #     for key in log_dict.keys()
        #     if key.endswith("loss") or key.endswith("error")
        # ]

        log_items = [
            [key, f"{log_dict[key]:.6f}"]
            for key in log_dict.keys()
            if not key.endswith("pred")
        ]

        message = tabulate(
            log_items,
            headers=[
                "Step: {:3d}".format(step),
                "Time: {:.3f}".format(end_time - start_time),
            ],
            tablefmt="github",
            numalign="right",
            disable_numparse=True,
        )

        message_len = len(message.split("\n")[0]) + 2
        seperator = "-" * message_len
        full_message = f"{seperator}\n{message}"

        for line in full_message.split("\n"):
            self.logger.info(line)
