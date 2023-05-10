from typing import Callable
from dataclasses import dataclass, field

from gaggle.base_nns.resnet_x import resnet20
from gaggle.base_nns.lenet import LeNet5
from gaggle.base_nns.snet import SNetCIFAR, SNetMNIST
from gaggle.base_nns.drqn import DRQN
from gaggle.base_nns.dqn import DQN


@dataclass
class IndividualArgs:
    """ Argument class that contains the arguments relating to individuals """

    CONFIG_KEY = "individual_args"

    individual_name: str = field(default="nn", metadata={
        "help": "Individual class to use",
    })
    
    param_lower_bound: float = field(default=None, metadata={
        "help": "lower bound of restricted parameter value, if None, no clipping is performed"
    })

    param_upper_bound: float = field(default=None, metadata={
        "help": "upper bound of restricted parameter value, if None, no clipping is performed"
    })
    
    individual_size: int = field(default=100, metadata={
        "help": "length of the parameter tensor for the basic NumpyIndividual and PytorchIndividual. This argument"
                "is irrelevant for other individuals (unless custom coded it to be so)"
    })

    model_name: str = field(default="lenet", metadata={
        "help": "name of the model architecture. Only relevant for neural network individuals. "
                "Please see the 'get_base_model' method of this class.",
    })

    resolution: int = field(default=32, metadata={
        "help": "input resolution of the model",
        "choices": [32]
    })

    base_model_weights: str = field(default=None, metadata={
        "help": "Path to pre-trained base model weights. Can be a path or a weights file"
                "from the cloud (see below for defaults). The base_model_weights only specify"
                "the weight initialization strategy for the base model, but these weights would"
                "be overwritten, if there is different data in the model file itself."
                ""
                "Example Mappings: "
                "   - resnet18: ResNet18_Weights.DEFAULT   "
                "   - resnet34: ResNet34_Weights.DEFAULT   "
                "   - resnet50: ResNet50_Weights.DEFAULT   ",
    })

    model_ckpt: str = field(default=None, metadata={
        "help": "path to the model's checkpoint"
    })

    random_init: bool = field(default=True, metadata={
        "help": "initialization process for the models"
    })

    models = {
        "resnet20": resnet20,
        "lenet": LeNet5,
        "snet_cifar": SNetCIFAR,
        "snet_mnist": SNetMNIST,
        "drqn": DRQN,
        "dqn": DQN
    }

    @classmethod
    def get_keys(cls):
        r"""Gets the list of available NN pre-built models that can be created

        Returns:
            list of strings of model names

        """
        return list(cls.models.keys())

    @classmethod
    def update(cls, key, model):
        r"""Add a new model to the list of models that can be created

        Args:
            key: model name that will be used as the dictionary lookup key
            model: model class object, it needs to not be already initialized

        """

        assert isinstance(model, Callable)
        cls.models[key] = model

    def get_base_model(self, *args, **kwargs):
        r"""Gets a base model as specified by the self.model_name field

        Args:
            *args: args that will get passed down to the model initialization
            **kwargs: kwargs that will get passed down to the model initialization

        Returns:
            NN model (nn.Module unless custom model was added using cls.update)

        """

        model = self.models.get(self.model_name, None)
        if model is None:
            raise ValueError(self.model_name)
        return model(*args, **kwargs)
