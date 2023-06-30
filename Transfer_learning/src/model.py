import torchvision
from torch import nn

class LoadModule(object):
    def __init__(self, model:str):
        self.name = model

    def __call__(self):
        """
        DenseNet121
        ResNet18
        """
        if self.name == "DenseNet121":
           densenet121 = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
           densenet121.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)
           return densenet121

        elif self.name == "ResNet18":
            resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            resnet18.fc = nn.Linear(in_features=512, out_features=2, bias=True)
            return resnet18
        