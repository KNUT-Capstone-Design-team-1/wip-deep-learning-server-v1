import io
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from PIL import Image

from efficientnet_pytorch import EfficientNet
import torch

from CRAFT_pytorch import file_utils
import numpy as np

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)

def detect_pill_shape():
    model_name = 'efficientnet-b7'

    image_size = EfficientNet.get_image_size(model_name)
    model = EfficientNet.from_pretrained(model_name, num_classes=2)

    model.load_state_dict(torch.load('./pill_shape_model/class2.pt'))
    model.eval()

    pill_folder = './pill_image/'

    image_list, _, _ = file_utils.get_files(pill_folder)
    try:
        image_path = image_list[0]
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            inputs = transform_image(image_bytes=image_bytes)

        outputs = model(inputs)
        _, predict = torch.max(outputs, 1)
        if int(predict.numpy()) == 0:
            pill_shape = "원형"
        else:
            pill_shape = "타원형"
        return pill_shape
    except:
        print("shape_classification error")
