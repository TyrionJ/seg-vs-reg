import os
import sys
import glob
import time
import utils
import torch
import losses
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from torch import optim, nn
from natsort import natsorted
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import datasets, trans
from data.data_utils import pkload
import models.TransMorph as TransMorph
from models.TransMorph import CONFIGS as CONFIGS_TM

warnings.filterwarnings('ignore', category=UserWarning)

TM_dir = '/remote-home/hejj/Data/runtime/SoR_folder/TransMorph'
dr = f'/remote-home/hejj/Data/runtime/SoR_folder/morph_data'


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(batch_size, loss_type, dataset, cont_training):
    atlas_dir = f'{dr}/atlas.pkl'
    train_dir = f'{dr}/Train/'
    val_dir = f'{dr}/Val/'

    to_dir = f'{TM_dir}/{dataset}'
    exp_dir = f'{to_dir}/experiments'
    log_dir = f'{to_dir}/logs'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    weights = [1, 1]  # loss weights
    exp_dir = f'{exp_dir}/{dataset}_{loss_type}_{weights[0]}_diffusion_{weights[1]}/'
    log_dir = f'{log_dir}/{dataset}_{loss_type}_{weights[0]}_diffusion_{weights[1]}/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(log_dir)

    best_dsc = 0
    lr = 0.0001  # learning rate
    epoch_start = 0
    max_epoch = 500  # max training epoch
    num_classes = 36

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph']
    config.img_size = pkload(os.path.join(train_dir, os.listdir(train_dir)[0]))[0].shape
    model = TransMorph.TransMorph(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    if cont_training and os.path.exists(f'{exp_dir}/latest.pth.tar'):
        check = torch.load(f'{exp_dir}/latest.pth.tar')
        epoch_start = check['epoch']
        best_dsc = check['best_dsc']
        model.load_state_dict(check['state_dict'])
        optimizer.load_state_dict(check['optimizer'])
        print(f'{exp_dir}/latest.pth.tar loaded!, best_dsc={best_dsc:.4f}')

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.Seg_norm(),  # rearrange segmentation label to 1 to 46
                                       trans.NumpyType((np.float32, np.int16))])

    train_set = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
    val_set = datasets.IXIBrainInferDataset(natsorted(glob.glob(val_dir + '*.pkl')), atlas_dir, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    criterions = [losses.NCC_vxm() if loss_type == 'ncc' else nn.MSELoss(),
                  losses.Grad3d(penalty='l2')]
    writer = SummaryWriter(log_dir=log_dir)
    print(f'Config Info: img_size={config.img_size}, dataset={dataset}, loss_type={loss_type}')

    print('\nTraining Starts')
    for epoch in range(epoch_start, max_epoch):
        '''
        Training
        '''
        start_time = time.time()
        loss_all = utils.AverageMeter()
        model.train()
        torch.cuda.empty_cache()
        with tqdm(desc=f'Train {epoch + 1}/{max_epoch}', total=len(train_loader)) as p:
            for data in train_loader:
                adjust_learning_rate(optimizer, epoch, max_epoch, lr)
                # atlas_img, patient_img
                x, y = [t.cuda() for t in data]

                x_in = torch.cat((x, y), dim=1)
                output = model(x_in)
                loss = 0
                loss_vals = []
                for n, loss_function in enumerate(criterions):
                    curr_loss = loss_function(output[n], y) * weights[n]
                    loss_vals.append(curr_loss)
                    loss += curr_loss
                loss_all.update(loss.item(), y.numel())
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sim, reg = loss_vals[0].item(), loss_vals[1].item()
                p.set_postfix(**{'1.Avg': '%.4f' % loss_all.avg,
                                 '2.Bat': '%.4f' % loss.item(),
                                 '3.Sim': '%.6f' % sim,
                                 '4.Reg': '%.6f' % reg})
                p.update()

        writer.add_scalar('Loss/train', loss_all.avg, epoch)

        '''
        Validation
        '''
        output = [0, 0]
        eval_dsc = utils.AverageMeter()
        torch.cuda.empty_cache()
        with torch.no_grad():
            model.eval()
            with tqdm(total=len(val_loader), desc='Validation') as p:
                for data in val_loader:
                    # atlas_img, patient_img, atlas_seg, patient_seg
                    x, y, x_seg, y_seg = [t.cuda() for t in data]
                    x_in = torch.cat((x, y), dim=1)
                    output = model(x_in)
                    def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                    dsc = utils.dice_val(def_out.long(), y_seg.long(), num_classes)
                    eval_dsc.update(dsc, 1)

                    p.set_postfix(**{'eval_dsc': '%.4f' % eval_dsc.mean})
                    p.update()

        print(f'Training loss: {loss_all.avg:.4f}')
        print(f'Validation dsc: {eval_dsc.mean:.06f}')
        print('DSC per Class:', f'{[f"{i:.03f}" for i in eval_dsc.avg]}'.replace('\'', ''))
        eval_dsc.mean > best_dsc and print(f'Eureka!!! Find best DSC: {eval_dsc.mean:.06f}')
        print(f'Epoch period: {(time.time() - start_time):.2f}s')
        print()

        best_dsc = max(eval_dsc.mean, best_dsc)
        check = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(check, save_dir=f'{exp_dir}/', filename=f'dsc{eval_dsc.mean:.3f}.pth.tar')
        save_checkpoint(check, save_dir=f'{exp_dir}/', filename=f'latest.pth.tar')
        writer.add_scalar('DSC/validate', eval_dsc.mean, epoch)

        grid_img = mk_grid_img(8, 1, config.img_size)
        def_grid = reg_model_bilin([grid_img.float(), output[1]])

        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
    writer.close()


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


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_checkpoint(state, save_dir, filename, max_model_num=4):
    torch.save(state, save_dir + filename)
    if 'latest' not in filename:
        model_lists = natsorted(glob.glob(save_dir + 'dsc*'))
        while len(model_lists) > max_model_num:
            os.remove(model_lists[0])
            model_lists = natsorted(glob.glob(save_dir + 'dsc*'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('-ds', type=str, default='Dataset005_InHouse-Sim',
                        choices=['Dataset001_OASIS', 'Dataset002_InHouse', 'Dataset004_OASIS-Sim', 'Dataset005_InHouse-Sim'])
    parser.add_argument('-b', type=int, default=1, help='batch size')
    parser.add_argument('-lt', type=str, default='ncc', choices=['mse', 'ncc'])
    parser.add_argument('--c', action='store_true', help='continue train')
    args = parser.parse_args()

    '''
    GPU configuration
    '''
    GPU_iden = args.d
    GPU_num = torch.cuda.device_count()
    print('-----------------------Proj Info---------------------------')
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    print(f'Currently using: {torch.cuda.get_device_name(GPU_iden)} ({GPU_iden})')
    print('If the GPU is available? ' + str(torch.cuda.is_available()))

    main(args.b, args.lt, args.ds, True)
