import io
import matplotlib.pyplot as plt
from pill_shape_model.cli_models import Shape_ResNet18
import torchvision.transforms as transforms
import torchvision
from PIL import Image

import torch


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    plt.imshow(image)
    plt.show()
    return my_transforms(image).unsqueeze(0)


model = Shape_ResNet18(2)

model.load_state_dict(torch.load('class2.pt'))
model.eval()

with open("test.png", 'rb') as f:
    image_bytes = f.read()
    inputs = transform_image(image_bytes=image_bytes)

outputs = model(inputs)
_, predict = torch.max(outputs, 1)

print(predict)
