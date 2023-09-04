import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import shutil
from data_loader import Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import numpy as np
import torch
import cv2
from scipy.ndimage.morphology import distance_transform_edt as edt


import argparse


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def replace_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)
        os.makedirs(dir)


class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # if not np.any(x):
        #     x[0][0] = 1.0
        # elif not np.any(y):
        #     y[0][0] = 1.0

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.percentile(distances[indexes], 95))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
            ), "Only binary channel supported"

        pred = (pred > 0.5).byte()
        target = (target > 0.5).byte()
        if torch.sum(pred) == 0:
            pred[0][0][0][0] = 1
            # print(pred)
            # print(torch.sum(pred))
        # print(pred.shape)
        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
            ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
            ).float()


        return torch.max(right_hd, left_hd)


def evaluate(pred, gt):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_binary = (pred >= 0.5).float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = (gt >= 0.5).float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        TP = torch.Tensor([1]).cuda()
        
    # IoU
    IoU = TP / (TP + FP + FN)
    # DICE
    DICE = 2 * IoU / (IoU + 1)
    hd = hd_metric.compute(pred, gt)
    return IoU.cpu().numpy(), DICE.cpu().numpy(), hd


def get_args_parser():
    parser = argparse.ArgumentParser('Segmentation prediction', add_help=False)
    parser.add_argument('--weight_path', default='/qingbo/us_seg/weights_new_tn3k/202308140253_unetpp_convmae_us280k_deblur_bs16_512_seed8_normTrue/355_dice_0.83271.pt', type=str,
                        help='')
    parser.add_argument('--save_dir', default='/qingbo/us_seg/weights_new_tn3k/202308140253_unetpp_convmae_us280k_deblur_bs16_512_seed8_normTrue/test_pred/', type=str,
                        help='')
    parser.add_argument('--datapath', default='/qingbo/data/us/tn3k_split/', type=str,
                        help='dataset path')

    parser.add_argument('--is_deblur', default=True, type=bool, help='is deblurring')

    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    hd_metric = HausdorffDistance()

    weight_path = args.weight_path

    model = torch.load(weight_path)
    model.eval()

    if args.is_deblur:
        x_test_dir = os.path.join( args.datapath, 'images_gaussian/test/')
    else: 
        x_test_dir = os.path.join( args.datapath, 'images/test/')
    y_test_dir = os.path.join( args.datapath, 'masks/test/')

    t_size = 512

    save_dir = args.save_dir
    replace_dir(save_dir)

    img_files_all = os.listdir(x_test_dir)
    dice_lst = []
    iou_lst = []
    hd_lst = []

    test_dataset = Dataset(x_test_dir, y_test_dir, t_size=t_size)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for image, mask_gt, name in test_dataloader:
        image = image.to("cuda")
        mask_gt = mask_gt.to("cuda")
        with torch.no_grad():
            mask_pred = model(image)
        iou, dice, hd = evaluate(mask_pred, mask_gt)
        hd = hd_metric.compute(mask_pred, mask_gt)
        hd = hd.numpy()
    
        dice_lst.append(dice)
        iou_lst.append(iou)
        hd_lst.append(hd)
        
        mask_pred_show = (mask_pred.squeeze().cpu().numpy())*255
        cv2.imwrite(save_dir + name[0], mask_pred_show)

        print (name[0], "dice:", dice, "HD95:", hd)


    print ("Average DICE:", np.average(dice_lst))
    print ("Average IoU:", np.average(iou_lst))
    print ("Average HD:", np.average(hd_lst))




