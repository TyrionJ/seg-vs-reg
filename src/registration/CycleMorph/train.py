import os
import glob
import utils
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from typing import Any
from natsort import natsorted
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

from data import datasets, trans
from models.cycleMorph_model import CONFIGS, CycleMorph

warnings.filterwarnings('ignore', category=UserWarning)


class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.avg: Any = 0
        self.val = 0
        self.mean = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.mean = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.mean = np.mean(self.avg)


def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)


def get_default(argss, var_name, default_settings, default_value, key_list):
    if (var_name in argss) and (argss.__dict__[var_name] != default_value):
        return
    v = default_settings
    for k in key_list:
        v = v[k]
    argss.__dict__[var_name] = v


def prepare_input(resolution):
    x = torch.FloatTensor(1, *resolution)
    y = torch.FloatTensor(1, *resolution)
    return dict(x=(x, y))


def main(device, dataset, dr):
    batch_size = 1
    atlas_file = f'{dr}/atlas.pkl'
    train_dir = f'{dr}/Train/'
    val_dir = f'{dr}/Val/'
    to_dir = f'/remote-home/hejj/Data/runtime/SoR_folder/CycleMorph/{dataset}'

    exp_dir = f'{to_dir}/experiments'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    best_dsc = 0
    lr = 0.0001
    epoch_start = 0
    max_epoch = 500
    cont_training = True
    num_classes = 36
    opt = CONFIGS['Cycle-Morph-v0']

    reg_model = utils.register_model(opt.inputSize, 'nearest')
    reg_model.to(device)
    reg_model_bilin = utils.register_model(opt.inputSize, 'bilinear')
    reg_model_bilin.to(device)
    model = CycleMorph()

    check = None
    if cont_training and os.path.exists(f'{exp_dir}/latest.pth'):
        check = torch.load(f'{exp_dir}/latest.pth')
        best_dsc = check['best_dsc']
        epoch_start = check['epoch']
        opt.lr = round(lr * np.power(1 - epoch_start / max_epoch, 0.9), 8)
        print(f'{exp_dir}/latest.pth loaded!, best_dsc={best_dsc:.4f}')
    else:
        opt.lr = lr
    opt.device = device
    model.initialize(opt)

    if check is not None:
        model.netG_A.load_state_dict(check['state_dict_A'])
        model.netG_B.load_state_dict(check['state_dict_B'])
        model.optimizer_.load_state_dict(check['optimizer'])

    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])

    train_set = datasets.TrainingDataset(glob.glob(train_dir + '*.pkl'), atlas_file, transforms=train_composed)
    val_set = datasets.ValidationDataset(glob.glob(val_dir + '*.pkl'), atlas_file, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    print('Training Starts')
    for epoch in range(epoch_start, max_epoch):
        ''' Training '''
        loss_all = AverageMeter()
        loss_net_a = AverageMeter()
        with tqdm(desc=f'Train {epoch + 1}/{max_epoch}', total=len(train_loader)) as p:
            for data in train_loader:
                # atlas_img, patient_img
                x, y = [t.to(device) for t in data[0:2]]
                model.set_input([x, y])
                loss_out, loss_reg, loss_net = model.optimize_parameters()
                loss_all.update(loss_out, y.numel())
                loss_net_a.update(loss_net, y.numel())

                p.set_postfix(**{'1.Avg': '%.4f' % loss_all.avg,
                                 '2.Bat': '%.4f' % loss_out,
                                 '3.NetA': '%.6f' % loss_net_a.avg,
                                 '4.Reg': '%.6f' % loss_reg})
                p.update()

        ''' Validation '''
        eval_dsc = AverageMeter()
        with torch.no_grad():
            with tqdm(desc=f'Valid {epoch + 1}/{max_epoch}', total=len(val_loader)) as p:
                for data in val_loader:
                    # atlas_img, patient_img, atlas_seg, patient_seg
                    x, y, x_seg, y_seg = [t.to(device) for t in data]
                    model.set_input([x, y])
                    model.test()
                    visuals = model.get_test_data()
                    flow = visuals['flow_A']
                    def_out = reg_model([x_seg.float(), flow])
                    dsc = utils.dice_val(def_out.long(), y_seg.long(), num_classes)
                    eval_dsc.update(dsc, 1)

                    p.set_postfix(**{'eval_dsc': '%.4f' % eval_dsc.mean})
                    p.update()

        print('Train avg los:', f'{loss_all.avg:.04f}')
        print('Valid avg DSC:', f'{eval_dsc.mean:.04f}')
        print('DSC per Class:', f'{[f"{i:.03f}" for i in eval_dsc.avg]}'.replace('\'', ''))

        check = {
            'epoch': epoch + 1,
            'state_dict_A': model.netG_A.state_dict(),
            'state_dict_B': model.netG_B.state_dict(),
            'best_dsc': max(eval_dsc.mean, best_dsc),
            'optimizer': model.optimizer_.state_dict(),
        }
        save_checkpoint(check, save_dir=f'{exp_dir}/', filename=f'dsc{eval_dsc.mean:.3f}.pth')
        save_checkpoint(check, save_dir=f'{exp_dir}/', filename=f'latest.pth')

        if eval_dsc.mean > best_dsc:
            best_dsc = eval_dsc.mean
            print(f'Eureka!!! Find best: epoch={epoch + 1}, dsc={best_dsc:.04f}')

        loss_all.reset()
        loss_net_a.reset()
        print()

    print('Training Done')


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


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
    parser.add_argument('-d', type=str, default='cpu', help='device: cpu or 0, 1, 2, ...')
    parser.add_argument('-ds', type=str, default='Dataset005_InHouse-Sim',
                        choices=['Dataset001_OASIS', 'Dataset002_InHouse', 'Dataset004_OASIS-Sim', 'Dataset005_InHouse-Sim'])
    args = parser.parse_args()

    d = torch.device('cpu') if args.d == 'cpu' else torch.device(f'cuda:{args.d}')
    _dr = f'/remote-home/hejj/Data/runtime/SoR_folder/morph_data/{args.ds}'
    main(d, args.ds, _dr)
