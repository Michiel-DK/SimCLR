from torchvision import models



def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "resnet50": models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]
