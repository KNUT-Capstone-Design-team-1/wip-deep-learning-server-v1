# DL Server Flask

- install python3, python3-pip
- require version == 3.8.1
- install pytorch
```
pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
- install requirements
```
pip3 install flask opencv-python scipy scikit-image lmdb natsort
```
- install EfficientNet-Pytorch
```
pip3 install efficientnet_pytorch
```

## Flask default
- port : 5000
- Only http, connect POST / return pill-feature json file
### Run flask in background
- if you want kill flask task in background
```
nohup python3 DL_main.py &
```
- check flask PID and kill
```
ps -ef
```

## Trained Model
- Pill-shape : [Click](https://drive.google.com/file/d/1pBHtpsecIVQptD3HoWt61gwBIgO2uHCR/view?usp=sharing)
<br>**Download file on pill_shape_model folder**
- Detect-text : [Click](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
<br>**Download file on CRAFT_pytorch/weights folder**
- Text-Recognition : [Click](https://drive.google.com/file/d/1yLixadZ_3Ls4x_TR0-8MG6-iQSEn5ZSG/view?usp=sharing)
<br>**Download file on deep_text_recognition folder**

## Requirements
- flask, pytorch, torchvision, opencv-python, scipy, scikit-image, pillow

## Test Enviroment
- CUDA == 10.1
- Python == 3.8.1 / Pytorch == 1.7.1 / torchvision == 0.8.2
- opencv-python == 4.4.0 / scipy == 1.4.1 / scikit-image == 0.18.1
- pillow == 8.0.1

## Links
- Repo of Text-Detection, CRAFT : https://github.com/clovaai/CRAFT-pytorch
- Repo of Text-Recognition : https://github.com/clovaai/deep-text-recognition-benchmark
- EfficientNet-Pytorch : https://github.com/lukemelas/EfficientNet-PyTorch