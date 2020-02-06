import torch
import argparse
import numpy as np
from model import vdsr
from data_utils import dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.io as sio

parser = argparse.ArgumentParser()
# dataset option
parser.add_argument('--test_truth_dir', default=r'./data/test_output')
parser.add_argument('--test_input_dir', default=r'./data/test_input')
parser.add_argument('--checkpoint_dir', default='checkpoints/')
parser.add_argument('--result_dir', default='result/')
# model option
parser.add_argument('--im_size', type=int, default=256)
parser.add_argument('--model', type=str, default='vdsr')
parser.add_argument('--input_nc', type=int, default=1)
parser.add_argument('--output_nc', type=int, default=9)
parser.add_argument('--suffix', type=str, default='9ch')
# other option
parser.add_argument('--test_epoch', type=int, default=45, help='test epoch')
parser.add_argument('--device', type=str, default='cuda:0', help='gpu name')
# define opt
opt = parser.parse_args()

model = vdsr.Net(opt).cuda()
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

    path = opt.result_dir + str(iter)
    pred_np = pred.detach().cpu()[0, :, :, :].numpy()
    sio.savemat(path, {'test': pred_np})

    pixel_loss = MSELoss(pred, truth)
    print('image no.%d, mse loss is %.3f' % (iter, pixel_loss))
