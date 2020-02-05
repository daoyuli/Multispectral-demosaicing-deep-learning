import os
import torch
from torch import nn
import argparse
import pandas as pd
from model import vdsr
from data_utils import dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
# dataset option
parser.add_argument('--train_truth_dir', default=r'./data/output_9(1-700)')
parser.add_argument('--train_input_dir', default=r'F:\PythonLab\SIM_pics\train\trainA')
parser.add_argument('--val_truth_dir', default=r'F:\PythonLab\SIM_pics\val\valB')
parser.add_argument('--val_input_dir', default=r'F:\PythonLab\SIM_pics\val\valA')
parser.add_argument('--checkpoint_dir', default='checkpoints/', help='checkpoint dir of model params')
parser.add_argument('--stat_dir', type=str, default='statistics/', help='statistics dir')
# model option
parser.add_argument('--im_size', type=int, default=256, help='size of image')
parser.add_argument('--model', type=str, default='vdsr')
parser.add_argument('--input_nc', type=int, default=1, help='input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=9, help='output image channels: 3 for RGB and 1 for grayscale')
# other option
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--start_epoch', type=int, default=0, help='strat epoch')
parser.add_argument('--epoch', type=int, default=500, help='total epoch')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--suffix', type=str, default='9ch', help='expriment name')
parser.add_argument('--device', type=str, default='cuda:0', help='')
# define opt
opt = parser.parse_args()

device = torch.device(opt.device)
model = vdsr.Net(opt)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0, 1])
else:
    model = nn.DataParallel(model, device_ids=[0])
model.to(device)
if opt.start_epoch > 0:
    model.load_state_dict(torch.load(opt.checkpoint_dir + '%s_epoch_%d_%s.pkl'
                                     % (opt.model, opt.start_epoch, opt.suffix)))
print('# network parameters:', sum(param.numel() for param in model.parameters()))

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, min_lr=1e-5)
# MAELoss = torch.nn.L1Loss()
MSELoss = torch.nn.MSELoss()

# load dataset
train_dataset = dataset(opt.train_input_dir, opt.train_truth_dir)
train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
val_dataset = dataset(opt.val_input_dir, opt.val_truth_dir)
val_data_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

results = {'p_loss': []}
results_val = {'p_loss': []}

for epoch in range(opt.epoch-opt.start_epoch):
    epoch_true = epoch + opt.start_epoch + 1
    # train model
    model.train()
    scheduler.step()

    p_loss = 0
    pic_num = 0

    learning_rate = optimizer.param_groups[0]['lr']
    print('epoch: %d learning rate: %.7f' % (epoch_true, learning_rate))

    for iter, train_data in enumerate(train_data_loader):
        input, truth = train_data
        input, truth = Variable(input, requires_grad=True).to(device), Variable(truth, requires_grad=True).to(device)
        batch_num = input.size(0)
        pic_num += batch_num
        
        optimizer.zero_grad()
        pred = model(input)
        loss = MSELoss(pred, truth)
        # back prop
        loss.backward()
        optimizer.step()

        # print loss statistics
        if iter % 100 == 0:
            print('training... epoch: %d, iter: %d, loss: %.4f' % (epoch_true, iter, loss))

        # print training results in every epoch
        p_loss += loss.data.item() * batch_num

    # save model params in every epoch
    if (epoch_true-1) % 5 == 0:
        torch.save(model.state_dict(), opt.checkpoint_dir + '%s_epoch_%d_%s.pkl'
                   % (opt.model, epoch_true-1, opt.suffix))

    results['p_loss'].append(p_loss / pic_num)
    data_frame = pd.DataFrame(
        data={'pixel loss': results['p_loss']},
        index=range(opt.start_epoch+1, epoch_true+1)
    )
    data_frame.to_csv(opt.stat_dir + 'train_results.csv', index_label='Epoch')

    # val model
    p_loss_val = 0
    pic_num_val = 0

    model.eval()
    for iter, val_data in enumerate(val_data_loader):
        input_val, truth_val = val_data
        input_val, truth_val = Variable(input_val, requires_grad=False).to(device), Variable(truth_val, requires_grad=False).to(device)

        batch_num_val = input_val.size(0)
        pic_num_val += batch_num_val

        pred_val = model(input_val)
        loss_val = MSELoss(pred_val, truth_val)
        p_loss_val += loss_val.data.item() * batch_num

    # print valing results in every epoch
    results_val['p_loss'].append(p_loss_val / pic_num_val)

    data_frame_val = pd.DataFrame(
        data={'pixel loss': results_val['p_loss']},
        index=range(opt.start_epoch+1, epoch_true+1)
    )
    data_frame_val.to_csv(opt.stat_dir + opt.model + 'val_results.csv', index_label='Epoch')
