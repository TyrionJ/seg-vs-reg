import os
import glob
import torch
import warnings
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from torch.optim import Adam
from natsort import natsorted
import torch.utils.data as Data

from utils import losses
from utils.config import args
from Models.TransMatch import TransMatch
from Models.STN import SpatialTransformer
from utils.datagenerators_atlas import Dataset

warnings.filterwarnings('ignore')


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))


def compute_label_dice(gt, pred, num_classes):
    cls_lst = list(range(1, num_classes))
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return dice_lst


def train():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    log_name = str(args.epochs) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

    f_img = sitk.ReadImage(args.template_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]

    input_fixed_eval = torch.from_numpy(input_fixed).to(device).float()
    input_fixed = np.repeat(input_fixed, args.batch_size, axis=0)
    input_fixed = torch.from_numpy(input_fixed).to(device).float()
    fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(args.atlas_file))[np.newaxis, np.newaxis, ...]
    fixed_label = torch.from_numpy(fixed_label).to(device).float()
    num_classes = int(fixed_label.max().cpu()) + 1

    net = TransMatch(args).to(device)

    best_dice = 0
    iterEpoch = 1
    checkpoint = None
    latest_model = f'{args.save_model_dir}/latest.pth.tar'
    if os.path.exists(latest_model):
        checkpoint = torch.load(latest_model)
        net.load_state_dict(checkpoint['state_dict'])
        iterEpoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['best_dice']
        print(f'Load checkpoint, epoch={iterEpoch}')

    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    net.train()
    STN.train()

    opt = Adam(net.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss
    if checkpoint is not None:
        opt.load_state_dict(checkpoint['optimizer'])

    # Get all the names of the training data
    train_files = glob.glob(os.path.join(args.train_dir, '*.nii.gz'))
    DS = Dataset(files=train_files)
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Training loop.
    for epoch in range(iterEpoch, args.epochs + 1):
        # Generate the moving images and convert them to tensors.
        net.train()
        STN.train()

        with tqdm(desc=f'[{epoch}/{args.epochs}]Training', total=len(DL)) as p:
            avg_loss, avg_sim, avg_grad = 0, 0, 0
            for N, (input_moving, fig_name) in enumerate(DL):
                # [B, C, D, W, H]
                fig_name = fig_name[0]
                input_moving = input_moving.to(device).float()

                # Run the data through the model to produce warp and flow field
                flow_m2f = net(input_fixed, input_moving)
                m2f = STN(input_fixed, flow_m2f)

                # Calculate loss
                sim_loss = sim_loss_fn(m2f, input_moving)
                grad_loss = grad_loss_fn(flow_m2f)
                # zero_loss = zero_loss_fn(flow_m2f, zero)
                loss = sim_loss + args.alpha * grad_loss  # + zero_loss

                avg_loss = (avg_loss * N + loss.item()) / (N + 1)
                avg_sim = (avg_sim * N + sim_loss.item()) / (N + 1)
                avg_grad = (avg_grad * N + grad_loss.item()) / (N + 1)
                print("%d, %s, %f, %f, %f" % (N, fig_name, loss.item(), sim_loss.item(), grad_loss.item()), file=f)

                # Backwards and optimize
                opt.zero_grad()
                loss.backward()
                opt.step()

                # inverse fixed image and moving image
                flow_m2f = net(input_moving, input_fixed)
                m2f = STN(input_moving, flow_m2f)

                # Calculate loss
                sim_loss = sim_loss_fn(m2f, input_fixed)
                grad_loss = grad_loss_fn(flow_m2f)
                # zero_loss = zero_loss_fn(flow_m2f, zero)
                loss = sim_loss + args.alpha * grad_loss  # + zero_loss

                print("%d, %s, %f, %f, %f" % (N, fig_name, loss.item(), sim_loss.item(), grad_loss.item()), file=f)

                # Backwards and optimize
                opt.zero_grad()
                loss.backward()
                opt.step()

                p.set_postfix(**{'1.loss': f'{avg_loss:.6f}',
                                 '2.sim': f'{avg_sim:.6f}',
                                 '3.grad': f'{grad_loss:.6f}'})
                p.update()

        test_file_lst = glob.glob(os.path.join(args.test_dir, "*.nii.gz"))

        net.eval()
        STN.eval()
        STN_label.eval()

        with tqdm(desc=f'[{epoch}/{args.epochs}]Validation', total=len(test_file_lst), colour='green') as p:
            dices_arr = []
            for N, file in enumerate(test_file_lst):
                name = os.path.split(file)[1]
                # 读入moving图像
                input_moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis, np.newaxis, ...]
                input_moving = torch.from_numpy(input_moving).to(device).float()
                # 读入moving图像对应的label
                input_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.label_dir, name)))

                # 获得配准后的图像和label
                pred_flow = net(input_fixed_eval, input_moving)
                # pred_img = STN(input_fixed_eval, pred_flow)
                pred_label = STN_label(fixed_label, pred_flow)
                # pred_label = input_label # 用于测试初始的dice值

                # 计算 DSC
                dices = compute_label_dice(input_label, pred_label[0, 0, ...].cpu().detach().numpy(), num_classes)
                dices_arr.append(dices)
                p.update()

        dices_arr = np.mean(dices_arr, axis=0)
        mean_fg_dice = np.nanmean(dices_arr)
        print(f'valid mean Dice: {np.round(mean_fg_dice, decimals=6)}')
        print(f'Dice per class: [{", ".join([f"{i:.3f}" for i in dices_arr])}]')
        if mean_fg_dice > best_dice:
            print(f'Eureka!!! Best dice: {mean_fg_dice:.4f}')
            best_dice = mean_fg_dice
        print()

        state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': opt.state_dict(),
            'best_dice': best_dice
        }
        save_checkpoint(state, args.save_model_dir, f'/dsc-{mean_fg_dice:.4f}_epoch-{epoch:03d}.pth.tar')
        torch.save(state, latest_model)

    f.close()


def save_checkpoint(state, save_dir, filename, max_model_num=4):
    model_lists = natsorted(glob.glob(save_dir + '/dsc-*.pth.tar'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, save_dir + filename)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
