from dataclasses import dataclass, field

import yaml

from gaggle.arguments.sys_args import SysArgs
from gaggle.arguments.individual_args import IndividualArgs
from gaggle.arguments.outdir_args import OutdirArgs
from gaggle.arguments.ga_args import GAArgs
from gaggle.arguments.problem_args import ProblemArgs
from gaggle.utils.special_print import print_warning
import transformers


def parse_args():
    """Helper function that parses the argument classes into a list of initialized argument objects with the given
    CLI argument values.

    Returns:
        Returns list of [OutdirArgs, SysArgs, IndividualArgs, GAArgs, ProblemArgs, ConfigArgs]
    """
    parser = transformers.HfArgumentParser((OutdirArgs, SysArgs, IndividualArgs, GAArgs, ProblemArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


@dataclass
class ConfigArgs:
    """ Argument class that allows to combine all the other arguments together and read 
    from config files for experiments"""

    config_path: str = field(default=None, metadata={
        "help": "path to the yaml configuration file (*.yml)"
    })

    def exists(self):
        return self.config_path is not None

    args_to_config = {  # specify the config keys to read in the *.yml file
        SysArgs.CONFIG_KEY: SysArgs(),
        IndividualArgs.CONFIG_KEY: IndividualArgs(),
        OutdirArgs.CONFIG_KEY: OutdirArgs(),
        GAArgs.CONFIG_KEY: GAArgs(),
        ProblemArgs.CONFIG_KEY: ProblemArgs()
    }

    @classmethod
    def get_keys(cls):
        """

        Returns: the list of config keys that will be read in the *.yml file

        """
        return list(cls.args_to_config.keys())

    @classmethod
    def update(cls, config_key, arg_subclass):
        r"""Add or replace one of the argument classes in the args_to_config that will 
        be read in the *.yml file.

        Args:
            config_key: key of the argument class to be added/replaced
            arg_subclass: argument class that will be called when using the given config_key

        Notes:
            arg_subclass needs to be an un-initialized object as the update will initialize it.

        """

        try:
            assert config_key in cls.args_to_config.keys()
        except AssertionError:
            print(f"Config Key {config_key} is not a valid config key")
            print(f"Valid Config Key: {list(cls.args_to_config.keys())}")
            return
        try:
            assert issubclass(arg_subclass, type(cls.args_to_config[config_key]))
        except AssertionError:
            print(f"Given class needs to be a subclass of the replaced arg")
            print(arg_subclass)
            print(type(cls.args_to_config[config_key]))
            return

        cls.args_to_config[config_key] = arg_subclass()

    def get_args(self):
        return self.get_outdir_args(), self.get_sys_args(), self.get_individual_args(), self.get_problem_args(), \
               self.get_ga_args()

    def get_sys_args(self) -> SysArgs:
        return self.args_to_config[SysArgs.CONFIG_KEY]

    def get_problem_args(self) -> ProblemArgs:
        return self.args_to_config[ProblemArgs.CONFIG_KEY]

    def get_individual_args(self) -> IndividualArgs:
        return self.args_to_config[IndividualArgs.CONFIG_KEY]

    def get_outdir_args(self) -> OutdirArgs:
        return self.args_to_config[OutdirArgs.CONFIG_KEY]

    def get_ga_args(self) -> GAArgs:
        return self.args_to_config[GAArgs.CONFIG_KEY]

    def __post_init__(self):
        if self.config_path is None:
            return

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        self.keys = list(data.keys())

        # load arguments
        keys_not_found = []
        for entry, values in data.items():
            for key, value in values.items():
                if key not in self.args_to_config[entry].__dict__.keys():
                    keys_not_found += [(entry, key)]
                self.args_to_config[entry].__dict__[key] = value
        if len(keys_not_found) > 0:
            print_warning(f"Could not find these keys: {keys_not_found}. Make sure they exist.")
