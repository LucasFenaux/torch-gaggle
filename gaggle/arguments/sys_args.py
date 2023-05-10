import os
from dataclasses import dataclass, field


@dataclass
class SysArgs:
    """ Argument class that contains the arguments relating to system settings and environment variables """

    CONFIG_KEY = "sys_args"

    num_workers: int = field(default=8, metadata={
        "help": "number of workers"
    })

    device: str = field(default="cuda", metadata={
        "help": "device to run algorithms on"
    })

    gpu_id: int = field(default=0, metadata={
        "help": "which gpu id to use for training"
    })

    use_dataloader: bool = field(default=False, metadata={
        "help": "Whether to load data using pytorch's dataloader object. This is highly discouraged as this is very "
                "inefficient for our purposes"
    })

    verbose: bool = field(default=True, metadata={
        "help": "whether to print out to the cmd line"
    })

    def set_gpu_id(self, gpu_id: int):
        assert self.device == "cuda"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
