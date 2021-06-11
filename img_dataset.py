import os
from PIL import Image
from torch.utils.data import Dataset

class ImgDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = root
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            # 이미지를 흑백으로 변환
            img = Image.fromarray(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # 이미지가 없을 경우 더미 이미지 생성
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])