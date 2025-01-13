import os
import sys
import glob
import torch
import random
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from torch import optim
from natsort import natsorted
from torchvision import transforms
from torch.utils.data import DataLoader

import utils
import losses
from models import RDP
from data import datasets, trans

warnings.filterwarnings('ignore', category=UserWarning)
data_root = f'/remote-home/hejj/Data/runtime/SoR_folder/morph_data/'

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


same_seeds(24)


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(device, dataset):
    dr = f'{data_root}/{dataset}'
    RDP_dir = '/remote-home/hejj/Data/runtime/SoR_folder/RDP'

    if not os.path.exists(RDP_dir):
        os.makedirs(RDP_dir)

    batch_size = 1
    atlas_file = f'{dr}/atlas.pkl'
    train_dir = f'{dr}/Train/'
    val_dir = f'{dr}/Val/'

    best_dsc = 0
    epoch_start = 0
    max_epoch = 300
    img_size = (160, 224, 160)
    cont_training = True
    weights = [1, 1]
    lr = 0.0001
    num_classes = 36

    save_dir = f'{dataset}_RDP_ncc-{weights[0]}_reg-{weights[1]}_lr-{lr}/'
    if not os.path.exists(f'{RDP_dir}/experiments/' + save_dir):
        os.makedirs(f'{RDP_dir}/experiments/' + save_dir)
    if not os.path.exists(f'{RDP_dir}/logs/' + save_dir):
        os.makedirs(f'{RDP_dir}/logs/' + save_dir)
    sys.stdout = Logger(f'{RDP_dir}/logs/' + save_dir)

    '''
    Initialize model
    '''
    model = RDP(img_size, channels=16)
    model.to(device)

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.to(device)

    '''
    If continue from previous training
    '''
    check = None
    if cont_training and os.path.exists(f'{RDP_dir}/experiments/{save_dir}latest.pth'):
        updated_lr = round(lr * np.power(1 - epoch_start / max_epoch, 0.9), 8)
        check = torch.load(f'{RDP_dir}/experiments/{save_dir}latest.pth')
        model.load_state_dict(check['state_dict'])
        best_dsc = check['best_dsc']
        epoch_start = check['epoch']
        print('Load check latest.pth, epoch =', epoch_start)
    else:
        updated_lr = lr

    ''' Initialize training '''
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])

    train_set = datasets.TrainingDataset(glob.glob(train_dir + '*.pkl'), atlas_file, transforms=train_composed)
    val_set = datasets.ValidationDataset(glob.glob(val_dir + '*.pkl'), atlas_file, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    if check is not None:
        optimizer.load_state_dict(check['optimizer'])
    criterions = [losses.NCC_vxm(), losses.Grad3d(penalty='l2')]

    print('Training Starts')
    for epoch in range(epoch_start, max_epoch):
        ''' Training '''
        loss_all = utils.AverageMeter()
        model.train()
        with tqdm(desc=f'Train {epoch + 1}/{max_epoch}', total=len(train_loader)) as p:
            avg_sim, avg_reg = 0, 0
            for N, data in enumerate(train_loader):
                adjust_learning_rate(optimizer, epoch, max_epoch, lr)

                # atlas_img, subject_img
                x, y = [t.to(device) for t in data]
                output = model(x, y)

                loss = 0
                loss_vals = []
                for n, loss_function in enumerate(criterions):
                    curr_loss = loss_function(output[n], y) * weights[n]
                    loss_vals.append(curr_loss)
                    loss += curr_loss
                loss_all.update(loss.item(), y.numel())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_sim = (avg_sim * N + loss_vals[0].item()) / (N + 1)
                p.set_postfix(**{'1.avg': '%.4f' % loss_all.avg,
                                 '2.bat': '%.4f' % loss.item(),
                                 '3.Sim': '%.6f' % avg_sim,
                                 '4.Reg': '%.6f' % avg_reg})
                p.update()

        ''' Validation '''
        eval_dsc = utils.AverageMeter()
        model.eval()
        with torch.no_grad():
            with tqdm(desc=f'Valid {epoch + 1}/{max_epoch}', total=len(val_loader)) as p:
                for data in val_loader:
                    # atlas_img, subject_img, atlas_seg, subject_seg
                    x, y, x_seg, y_seg = [t.to(device) for t in data]
                    output = model(x, y)

                    def_out = reg_model([x_seg.float(), output[1]])
                    dsc = utils.dice_val(def_out.long(), y_seg.long(), num_classes)
                    eval_dsc.update(dsc, x.size(0))

                    p.set_postfix(**{'eval_dsc': '%.4f' % eval_dsc.mean})
                    p.update()

        print('Train avg los:', f'{loss_all.avg:.04f}')
        print('Valid avg DSC:', f'{eval_dsc.mean:.04f}')
        print('DSC per Class:', f'{[f"{i:.03f}" for i in eval_dsc.avg]}'.replace('\'', ''))

        check = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': max(eval_dsc.mean, best_dsc),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(check, save_dir=f'{RDP_dir}/experiments/{save_dir}', filename=f'dsc{eval_dsc.mean:.4f}.pth')
        save_checkpoint(check, save_dir=f'{RDP_dir}/experiments/{save_dir}', filename=f'latest.pth')
        if eval_dsc.mean > best_dsc:
            best_dsc = eval_dsc.mean
            print(f'Eureka!!! Find best: epoch={epoch + 1}, dsc={best_dsc:.04f}')

        loss_all.reset()
        print()


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - epoch / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir, filename, max_model_num=4):
    torch.save(state, save_dir + filename)
    if 'latest' not in filename:
        model_lists = natsorted(glob.glob(save_dir + 'dsc*'))
        while len(model_lists) > max_model_num:
            os.remove(model_lists[0])
            model_lists = natsorted(glob.glob(save_dir + 'dsc*'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='0', help='device: cpu or 0, 1, 2, ...')
    parser.add_argument('-ds', type=str, default='Dataset005_InHouse-Sim',
                        choices=['Dataset001_OASIS', 'Dataset002_InHouse', 'Dataset004_OASIS-Sim', 'Dataset005_InHouse-Sim'])
    args = parser.parse_args()

    d = torch.device('cpu') if args.d == 'cpu' else torch.device(f'cuda:{args.d}')
    main(d, args.ds)
