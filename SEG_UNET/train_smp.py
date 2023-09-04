import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch.nn as nn

from data_loader import Dataset
import augumentations as augu

import plots
import shutil
from sklearn.metrics import f1_score, jaccard_score
import argparse
import random

import datetime



def set_global_random_seed(seed):
    os.environ['PYTHONASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark = False

    

def get_args_parser():
    parser = argparse.ArgumentParser('Segmentation', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size')
    # unet, unetpp
    parser.add_argument('--model', default='unetpp', type=str,
                        help='segmentation model (unet, unetpp)')
    parser.add_argument('--encoder', default='dconvmae', type=str,
                        help='encoder (convmae or dconvmae)')
    parser.add_argument('--encoder_weights', default='/qingbo/ConvMAE-main/pretrain_zk/0425_dconvmae_base_gaussian/checkpoint-799.pth', type=str,
                        help='encoder weights')
    parser.add_argument('--is_deblurring', default=True, type=bool,
                        help='is deblurring, should be consistent with encoder_weights')
    parser.add_argument('--input_size', default=512, type=int,
                        help='images input size')
    parser.add_argument('--datapath', default='/qingbo/data/us/tn3k_split/', type=str,
                        help='dataset path')
    parser.add_argument('--seed', default=0, type=int,
                        help='random seed')
    parser.add_argument('--output_dir', default='/qingbo/us_seg/weights_new_tn3k/', type=str,
                        help='output directory')

    return parser



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    set_global_random_seed(args.seed)
    
    DATA_DIR = args.datapath
   
    if  args.is_deblurring:
        x_train_dir = os.path.join(DATA_DIR, 'images_gaussian/train/')
        x_valid_dir = os.path.join(DATA_DIR, 'images_gaussian/val/')
    else: 
        x_train_dir = os.path.join(DATA_DIR, 'images/train/')
        x_valid_dir = os.path.join(DATA_DIR, 'images/val/')
    y_train_dir = os.path.join(DATA_DIR, 'masks/train/')
    y_valid_dir = os.path.join(DATA_DIR, 'masks/val/')


    ENCODER = args.encoder
    ENCODER_WEIGHTS = args.encoder_weights
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'
    n_class = 1
    batch_size = args.batch_size
    in_chans = 3
    is_rgb =True

# create segmentation model with pretrained encoder
    if args.model == "unet":
        model = smp.Unet( encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, 
                         in_channels=in_chans, classes=n_class, activation=ACTIVATION)
    elif args.model == "unetpp":
        model = smp.UnetPlusPlus( encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, 
                                 in_channels=in_chans, classes=n_class, activation=ACTIVATION)

    if torch.cuda.is_available():
        print ("CUDA is available, using GPU.")
        num_gpu = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=num_gpu)

    t_size = args.input_size
    if ENCODER_WEIGHTS == None:
        ENCODER_WEIGHTS = 'None'
    model_dir = args.output_dir + "/" + datetime.datetime.now().strftime('%Y%m%d%H%M_') + args.model + "_" + ENCODER + \
        "_deblur" + str(args.is_deblurring) + "_bs" + str(batch_size) + "_"  + str(t_size) + "_seed" + str(args.seed) + "/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_dataset = Dataset( x_train_dir, y_train_dir, augmentation=augu.get_training_augmentation(), t_size=t_size )

    valid_dataset = Dataset( x_valid_dir, y_valid_dir, t_size=t_size )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=8)

    print ("Train:", len(train_dataset))
    print ("Valid:", len(valid_dataset))

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.Fscore(),
        smp.utils.metrics.IoU(),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=1e-4),
    ])

    from segmentation_models_pytorch import utils
    train_epoch = utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    max_dice = 0
    best_epoch = 0
    EARLY_STOPS = 100
    train_dict = {'loss': [], 'dice': [], 'iou': [] }
    val_dict = {'loss': [], 'dice': [], 'iou': [] }

    for i in range(0, 100000):

        print('\nEpoch: {}'.format(i))
        print ("Best epoch:", best_epoch, "\tDICE:", max_dice)
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        train_dict['loss'].append(train_logs['dice_loss'])  # 'dice_loss + jaccard_loss'
        train_dict['dice'].append(train_logs['fscore'])
        train_dict['iou'].append(train_logs['iou_score'])
        val_dict['loss'].append(valid_logs['dice_loss'])
        val_dict['dice'].append(valid_logs['fscore'])
        val_dict['iou'].append(valid_logs['iou_score'])

        plots.save_loss_dice(train_dict, val_dict, model_dir)

        # do something (save model, change lr, etc.)
        if max_dice < valid_logs['fscore']:
            if max_dice != 0:
                old_filepath = model_dir + str(best_epoch) + "_dice_" + str(max_dice) + ".pt"
                os.remove(old_filepath)

            max_dice = np.round(valid_logs['fscore'], 5)
            torch.save(model, model_dir + str(i) + "_dice_" + str(max_dice) + ".pt")
            print('Model saved!')
            best_epoch = i


        if i - best_epoch > EARLY_STOPS:
            print (str(EARLY_STOPS), "epoches didn't improve, early stop.")
            print ("Best dice:", max_dice)
            break


