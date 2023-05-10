import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from gaggle.ga import GA
from gaggle.ga.ga_factory import GAFactory
from gaggle.arguments.config_args import parse_args


def train():
    """ Train a model from scratch on a data. """
    outdir_args, sys_args, individual_args, problem_args, ga_args, config_args = parse_args()
    if config_args.exists():
        outdir_args, sys_args, individual_args, problem_args, ga_args = config_args.get_args()

    trainer: GA = GAFactory.from_ga_args(ga_args=ga_args, problem_args=problem_args, sys_args=sys_args,
                                         outdir_args=outdir_args, individual_args=individual_args)
    trainer.train()


if __name__ == "__main__":
    train()
