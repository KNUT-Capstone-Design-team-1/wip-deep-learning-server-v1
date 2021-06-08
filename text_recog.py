import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from img_dataset import ImgDataset

from deep_text_recognition.utils import CTCLabelConverter, AttnLabelConverter
from deep_text_recognition.dataset import RawDataset, AlignCollate
from deep_text_recognition.model import Model

def recog_net(opt, device, img_list):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_img = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    # use RawDataset
    image_data = ImgDataset(root=img_list, opt=opt)
    image_loader = torch.utils.data.DataLoader(
        image_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_img, pin_memory=True)

    pred_text = []
    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in image_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                pred_text.append(pred)
    return pred_text

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', required=False, help='path to image_folder which contains text images')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
parser.add_argument('--saved_model', default='deep_text_recognition/pill_recog_model.pth', required=False, help="path to saved_model to evaluation")
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', default='TPS', type=str, required=False, help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', default='ResNet', type=str, required=False, help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', default='BiLSTM', type=str, required=False, help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', default='Attn', type=str, required=False, help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

def img_text_recog(img_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = parser.parse_args()

    """ vocab / character number configuration """
    opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    opt.num_gpu = torch.cuda.device_count()

    pill_text = recog_net(opt, device, img_list)
    pill_text.reverse()
    pill_text = "".join(pill_text)
    return pill_text
