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
        
        elif self.name == "DenseNet161":
           densenet161 = torchvision.models.densenet161(weights=torchvision.models.DenseNet161_Weights.IMAGENET1K_V1)
           densenet161.classifier = nn.Linear(in_features=densenet161.classifier.in_features, out_features=2, bias=True)
           return densenet161
        
        elif self.name == "DenseNet201":
           densenet201 = torchvision.models.densenet201(weights=torchvision.models.DenseNet201_Weights.IMAGENET1K_V1)
           densenet201.classifier = nn.Linear(in_features=densenet201.classifier.in_features, out_features=2, bias=True)
           return densenet201

        elif self.name == "ResNet18":
            resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            resnet18.fc = nn.Linear(in_features=512, out_features=2, bias=True)
            return resnet18
        
        elif self.name == "ResNet34":
            resnet34 = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
            resnet34.fc = nn.Linear(in_features=512, out_features=2, bias=True)
            return resnet34
        
        elif self.name == "ResNet50":
            resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            resnet50.fc = nn.Linear(in_features=resnet50.fc.in_features, out_features=2, bias=True)
            return resnet50
        
        elif self.name == "Inceptionv3":
            inceptionv3 = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
            inceptionv3.fc = nn.Linear(in_features=inceptionv3.fc.in_features, out_features=2, bias=True)
            return inceptionv3
        
        elif self.name == "AlexNet":
            alexnet = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
            alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)
            return alexnet
        