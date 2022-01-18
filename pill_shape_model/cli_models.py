import torch
import torchvision


class Shape_Effi_B7(torch.nn.Module):
    def __init__(self, class_num):
        super(Shape_Effi_B7, self).__init__()
        self.effi_b7 = torchvision.models.efficientnet_b7(pretrained=True)
        self.fc = torch.nn.Linear(1000, class_num)

    def forward(self, x):
        x = self.effi_b7(x)
        x = self.fc(x)

        return x


class Shape_Effi_B0(torch.nn.Module):
    def __init__(self, class_num):
        super(Shape_Effi_B0, self).__init__()
        self.effi_b0 = torchvision.models.efficientnet_b0(pretrained=True)
        self.fc = torch.nn.Linear(1000, class_num)

    def forward(self, x):
        x = self.effi_b0(x)
        x = self.fc(x)

        return x


class Shape_ResNet152(torch.nn.Module):
    def __init__(self, class_num):
        super(Shape_ResNet152, self).__init__()
        self.resnet152 = torchvision.models.resnet152(pretrained=True)
        self.fc = torch.nn.Linear(1000, class_num)

    def forward(self, x):
        x = self.resnet152(x)
        x = self.fc(x)

        return x


class Shape_ResNet18(torch.nn.Module):
    def __init__(self, class_num):
        super(Shape_ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=False)
        self.fc = torch.nn.Linear(1000, class_num)

    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc(x)

        return x


class Shape_VGG19(torch.nn.Module):
    def __init__(self, class_num):
        super(Shape_VGG19, self).__init__()
        self.resnet18 = torchvision.models.vgg19(pretrained=True)
        self.fc = torch.nn.Linear(1000, class_num)

    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc(x)

        return x
