# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import mmrotate  # noqa: F401


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img',
                        default='/media/dell/disk8/qaz/dataset/ms_dota1024/train/images/P0458__1024__324___131.png',
                        help='Image file')
    parser.add_argument('--config',
                        default='/media/dell/disk8/qaz/my_1/configs/group_ld_rotated_fcos/rotated_fcos_r18_34_50_101_fpn_1x_dota_le90_t_3.py',
                        help='Config file')
    parser.add_argument('--checkpoint',
                        default='/media/dell/disk8/qaz/my_1/tools/work_dirs/rotated_fcos_r18_34_50_101_fpn_1x_dota_le90_t_3/ang_im_max_14_680.pth',
                        help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
