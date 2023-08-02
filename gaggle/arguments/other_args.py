from dataclasses import dataclass


@dataclass
class OtherArgs:
    """ Argument class that contains any other arguments that could be needed and allows for additional subsequent
     arguments to be later added"""

    CONFIG_KEY = "other_args"
