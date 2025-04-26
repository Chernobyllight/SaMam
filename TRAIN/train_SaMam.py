import argparse
from argparse import ArgumentParser
from pathlib import Path

import os

project_root = os.path.abspath('..')
import sys

sys.path.append(project_root)

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from TRAIN.lightning_module import dataset
from TRAIN.lightning_module.datamodule import DataModule
from TRAIN.lightning_module.lightningmodel import LightningModel


# torch.backends.cudnn.enabled = False

class TensorBoardImageLogger(TensorBoardLogger):
    """
    Wrapper for TensorBoardLogger which logs images to disk,
        instead of the TensorBoard log file.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        exp = self.experiment

        # if not hasattr(exp, 'add_image'):
        exp.add_image = self.add_image

    def add_image(self, tag, img_tensor, global_step):
        dir = Path(self.log_dir, 'images')
        dir.mkdir(parents=True, exist_ok=True)

        file = dir.joinpath(f'{tag}_{global_step:09}.jpg')
        dataset.save(img_tensor, file)


def parse_args():
    # Init parser
    parser = ArgumentParser()

    # train_model setting
    parser.add_argument('--gpus', nargs='+', default='0',
                        help='GPU for training. Command usage sample: --gpus 0 1')
    parser.add_argument('--iterations', type=int, default=200_000,
                        help='The number of training iterations.')
    parser.add_argument('--log-dir', type=str, default='./',
                        help='The directory where the logs are saved to.')
    parser.add_argument('--checkpoint', type=str,
                        default=None,
                        help='Resume training from a checkpoint file.')
    parser.add_argument('--val-interval', type=int, default=1000,
                        help='How often a validation step is performed. '
                             'Applies the model to several fixed images and calculate the losses.')

    # dataset setting
    # you need to set your own training content and style datasets manually
    parser.add_argument('--content', type=str, default='/home/liuhd/dataset_ST/train2017',
                        help='Directory with content images.')
    parser.add_argument('--style', type=str, default='/home/liuhd/dataset_ST/train',
                        help='Directory with style images.')
    # test data during training
    parser.add_argument('--test-content', type=str, default='./test_images/content',
                        help='Directory with test content images. If not set, takes 5 random train_model content images.')
    parser.add_argument('--test-style', type=str, default='./test_images/style',
                        help='Directory with test style images. If not set, takes 5 random train_model style images.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size.')

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

    # training setting
    # Losses
    parser.add_argument('--style-weight', type=float, default=7.0)
    parser.add_argument('--content-weight', type=float, default=7.0)
    parser.add_argument('--if-identity-loss', type=int, default=1)
    parser.add_argument('--lambda1', type=float, default=70.0)
    parser.add_argument('--lambda2', type=float, default=1.0)
    # Optimizer
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    gpus = []
    for i in args['gpus']:
        gpus.append(int(i))

    if args['checkpoint'] is None:
        max_epochs = 1
        model = LightningModel(**args)
    else:
        # We need to increment the max_epoch variable, because PyTorch Lightning will
        # resume training from the beginning of the next epoch if resuming from a mid-epoch checkpoint.
        max_epochs = torch.load(args['checkpoint'])['epoch'] + 1
        model = LightningModel.load_from_checkpoint(checkpoint_path=args['checkpoint'])

    datamodule = DataModule(**args)

    logger = TensorBoardImageLogger(args['log_dir'], name='logs')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        accelerator="gpu",
        devices=gpus,
        max_epochs=max_epochs,
        max_steps=args['iterations'],
        val_check_interval=args['val_interval'],
        logger=logger,
        callbacks=[lr_monitor]
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint("./final_model.ckpt")
