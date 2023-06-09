import os
import shutil
import cv2
import argparse
import numpy as np

import albumentations as albu


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def replace_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)
        os.makedirs(dir)


def get_args_parser():
    parser = argparse.ArgumentParser('Image Blurring', add_help=False)
    parser.add_argument('--src_dir', default="", type=str,
                        help='source directory for images needed to be blurred.')
    parser.add_argument('--dst_dir', default="", type=str,
                        help='destination directory for blurred images to be saved.')
    parser.add_argument('--method', default='gaussian', type=str,
                        help='blurring method (gaussian, mean, median, motion, defocus).')
    parser.add_argument('--sigma', default=1.1, type=float,
                        help='blurriness')
    
    parser.add_argument('--seed', default=0, type=int)
    return parser
    
    
def main(args):
    np.random.seed(args.seed)
    
    make_dir(args.dst_dir)
    for dir_1 in os.listdir(args.src_dir):
        make_dir(os.path.join(args.dst_dir, dir_1))
        for dir_2 in os.listdir(os.path.join(args.src_dir, dir_1)):
            make_dir(os.path.join(args.dst_dir, dir_1, dir_2))
            for file_name in os.listdir(os.path.join(args.src_dir, dir_1, dir_2)):
                img = cv2.imread(os.path.join(args.src_dir, dir_1, dir_2, file_name),
                                 cv2.IMREAD_GRAYSCALE)
                if args.method == "gaussian":
                    img_blured = cv2.GaussianBlur(img, (0, 0), args.sigma)
                elif args.method == "mean":
                    img_blured = cv2.blur(img, (5, 5))
                elif args.method == "median":
                    img_blured = cv2.medianBlur(img, (5, 5))
                elif args.method == "motion":
                    transform = albu.Compose([
                        albu.MotionBlur(blur_limit=(5, 5), p=1), ], p=1)
                    img_blured = transform(image=img)['image']
                elif args.method == "defocus":
                    transform = albu.Compose([
                        albu.Defocus(radius=(5, 5), alias_blur=(0.01, 0.01),
                            always_apply=True, p=1),], p=1)
                    img_blured = transform(image=img)['image']

                cv2.imwrite(os.path.join(args.dst_dir, dir_1, dir_2, file_name), img_blured)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
