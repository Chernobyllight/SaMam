import argparse
from argparse import ArgumentParser
from pathlib import Path

import os

project_root = os.path.abspath('..')
import sys

sys.path.append(project_root)

import torch

from TEST import test_utils
from TRAIN.lightning_module.lightningmodel import LightningModel
from MODEL.SaMam_model import SaMam
from tqdm import tqdm


def stylize_image(model, content_file, style_file, style_size):
    device = next(model.parameters()).device

    content = test_utils.load(content_file)
    style = test_utils.load(style_file)

    content = test_utils.content_transforms()(content)
    style = test_utils.style_transforms(style_size)(style)

    content = content.to(device).unsqueeze(0)
    style = style.to(device).unsqueeze(0)

    output = model.forward(content, style)
    return output[0].detach().cpu()


def parse_args():
    # Init parser
    parser = ArgumentParser()

    # test dataset setting
    parser.add_argument('--content-dir', type=str, default='./test_images/content/',
                        help='Directory with test content images. If not set, takes 5 random train_model content images.')
    parser.add_argument('--style-dir', type=str, default='./test_images/style/',
                        help='Directory with test style images. If not set, takes 5 random train_model style images.')
    parser.add_argument('--output-dir', type=str, default='./test_images/output')
    parser.add_argument('--model_ckpt', type=str, default='/home/liuhd/SaMam_upload/checkpoint/iteration_200000.ckpt')
    parser.add_argument('--save-as', type=str, default='png')
    parser.add_argument('--style-size', type=int, default=256,
                        help='Style images are resized such that the smaller edge has this size. Any size is allowed')

    # model setting
    parser.add_argument('--nVSSMs', type=int, default=2)
    parser.add_argument('--nSAVSSMs', type=int, default=2)
    parser.add_argument('--nSAVSSGs', type=int, default=2)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--patch-size', type=int, default=8)
    parser.add_argument('--representation-dim', type=int, default=64)
    parser.add_argument('--d-state', type=int, default=16)
    parser.add_argument('--expand', type=float, default=2.0)
    parser.add_argument('--compress-ratio', type=int, default=8)
    parser.add_argument('--squeeze-factor', type=int, default=8)
    parser.add_argument('--mamba-from-trion', type=int, default=1)

    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    print(args)

    ext = args['save_as']

    content_files = test_utils.files_in(args['content_dir'])
    style_files = test_utils.files_in(args['style_dir'])
    output_dir = Path(args['output_dir'])
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if args['mamba_from_trion'] == 0:
        checkpoint1 = torch.load(args['model_ckpt'], map_location='cuda:0')
        SaMam_model = LightningModel(**args)
        keys = list(checkpoint1['state_dict'].keys())
        for k in keys:
            if k.startswith('loss_func'):
                del checkpoint1['state_dict'][k]
            else:
                new_k = k[6:]
                checkpoint1['state_dict'][new_k] = checkpoint1['state_dict'].pop(k)
        SaMam_model.model.load_state_dict(checkpoint1['state_dict'])

    else:
        SaMam_model = LightningModel.load_from_checkpoint(checkpoint_path=args['model_ckpt'], map_location='cuda:0')

    SaMam_model = SaMam_model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    SaMam_model.eval()

    pbar = tqdm(total=len(content_files) * len(style_files))
    with torch.no_grad():
        # Add style images at top row
        for i, content in enumerate(content_files):
            for j, style in enumerate(style_files):
                # Stylize content-style pair
                output = stylize_image(SaMam_model, content, style, style_size=args['style_size'])
                test_utils.save(output, output_dir.joinpath(f'{i:02}--{j:02}.{ext}'))
                pbar.update(1)
