from .lenet import lenet
from .vgg import vgg11_bn
from .resnet import resnet18

__factory__ = {
    "vgg": vgg11_bn,
    "lenet": lenet,
    "resnet": resnet18
}


def get_model(name):
    if name not in __factory__:
        raise ValueError("only support: {}".format(__factory__.keys()))
    return __factory__[name]()
