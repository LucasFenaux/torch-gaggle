from dataclasses import dataclass, field

import yaml

from gaggle.arguments.sys_args import SysArgs
from gaggle.arguments.individual_args import IndividualArgs
from gaggle.arguments.outdir_args import OutdirArgs
from gaggle.arguments.ga_args import GAArgs
from gaggle.arguments.problem_args import ProblemArgs
from gaggle.utils.special_print import print_warning
import transformers
from gaggle.other_args import OtherArgs


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
        ProblemArgs.CONFIG_KEY: ProblemArgs(),
        OtherArgs.CONFIG_KEY: OtherArgs()
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
               self.get_ga_args(), self.get_other_args()

    def get_other_args(self) -> OtherArgs:
        return self.args_to_config[OtherArgs.CONFIG_KEY]

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


def parse_args(outdir_args_cls=OutdirArgs, sys_args_cls=SysArgs, individual_args_cls=IndividualArgs,
               ga_args_cls=GAArgs, problem_args_cls=ProblemArgs, config_args_cls=ConfigArgs, other_args_cls=OtherArgs):
    """Helper function that parses the argument classes into a list of initialized argument objects with the given
    CLI argument values.

    Args:
        outdir_args_cls: class that takes care of the Outdir args behavior (needs to be a subclass of OutdirArgs)
        sys_args_cls: class that takes care of the Sys args behavior (needs to be a subclass of SysArgs)
        individual_args_cls: class that takes care of the Individual args behavior (needs to be a subclass of
        IndividualArgs)
        ga_args_cls: class that takes care of the GA args behavior (needs to be a subclass of GAArgs)
        problem_args_cls: class that takes care of the Problem args behavior (needs to be a subclass of ProblemArgs)
        config_args_cls: class that takes care of the Config args behavior (needs to be a subclass of ConfigArgs)
        other_args_cls: class that takes care of the Other args behavior (needs to be a subclass of OtherArgs)

    Returns:
        Returns list of [OutdirArgs, SysArgs, IndividualArgs, GAArgs, ProblemArgs, OtherArgs, ConfigArgs]

    """
    assert issubclass(outdir_args_cls, OutdirArgs)
    assert issubclass(sys_args_cls, SysArgs)
    assert issubclass(individual_args_cls, IndividualArgs)
    assert issubclass(ga_args_cls, GAArgs)
    assert issubclass(problem_args_cls, ProblemArgs)
    assert issubclass(config_args_cls, ConfigArgs)
    assert issubclass(other_args_cls, OtherArgs)

    parser = transformers.HfArgumentParser((outdir_args_cls, sys_args_cls, individual_args_cls,
                                            problem_args_cls, ga_args_cls, other_args_cls, config_args_cls))
    return parser.parse_args_into_dataclasses()
