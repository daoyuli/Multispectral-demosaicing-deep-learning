import torch
import argparse
import numpy as np
from model import vdsr
from data_utils import dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

parser = argparse.ArgumentParser()
# dataset option
parser.add_argument('--test_truth_dir', default=r'E:\LDY\SIM_pics\test\testB')
parser.add_argument('--test_input_dir', default=r'E:\LDY\SIM_pics\test\testA')
parser.add_argument('--checkpoint_dir', default='checkpoints/')
# model option
parser.add_argument('--im_size', type=int, default=256)
parser.add_argument('--model', type=str, default='vdsr')
parser.add_argument('--input_nc', type=int, default=1)
parser.add_argument('--output_nc', type=int, default=9)
parser.add_argument('--suffix', type=str, default='9ch')
# other option
parser.add_argument('--test_epoch', type=int, default=25, help='test epoch')
parser.add_argument('--device', type=str, default='cuda:0', help='gpu name')
# define opt
opt = parser.parse_args()

model = vdsr.vdsr(opt).cuda()
model.load_state_dict(torch.load(opt.checkpoint_dir + '%s_epoch_%d_%s.pkl'
                                 % (opt.model, opt.test_epoch, opt.suffix)))
MSELoss = torch.nn.MSELoss()

test_dataset = dataset(opt.test_input_dir, opt.test_truth_dir)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

total_pl = 0
pic_num = 0
model.eval()
print('test results of %s_%s_epoch_%s' % (opt.model, opt.test_epoch, opt.suffix))
for iter, test_data in enumerate(test_data_loader):
    input, truth = test_data
    input, truth = Variable(input, requires_grad=False).cuda(), Variable(truth, requires_grad=False).cuda()
    batch_num = input.size(0)
    pic_num += batch_num

    pred = model(input)

    pixel_loss = MSELoss(pred, truth)
    PSNR = 10*np.log10(1/pixel_loss.data.item())

    print('pic no.%d PSNR: %.4f' % (pic_num, PSNR))

    img_name = str(pic_num)
    out_img = ToPILImage()(pred[0].data.cpu())
    out_img.save('test_pics/'+img_name+'pred.png')
    truth_img = ToPILImage()(truth[0].data.cpu())
    truth_img.save('test_pics/'+img_name+'truth.png')

    total_pl += pixel_loss.data.item()

test_pl = total_pl/pic_num
print('average pixel loss: %.4f' % test_pl)
