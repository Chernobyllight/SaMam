from argparse import ArgumentParser
from math import sqrt
from statistics import mean

import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid

from MODEL.SaMam_model import SaMam
from LOSS.losses import Integration_loss


class LightningModel(pl.LightningModule):

    def __init__(self,
                 # model setting
                 nVSSMs, nSAVSSMs, nSAVSSGs,
                 embed_dim, patch_size, representation_dim, d_state, expand, compress_ratio, squeeze_factor,
                 mamba_from_trion,

                 # loss setting
                 style_weight=7.0,
                 content_weight=7.0,
                 lambda1=70.0, lambda2=1.0,

                 lr=0.0001,
                 **_):
        super().__init__()
        self.validation_step_outputs = []
        self.save_hyperparameters()

        self.lr = lr
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # Style loss
        self.loss_func = Integration_loss()

        # Model
        self.model = SaMam(
            nVSSMs=nVSSMs,
            nSAVSSMs=nSAVSSMs,
            nSAVSSGs=nSAVSSGs,

            embed_dim=embed_dim,
            patch_size=patch_size,
            representation_dim=representation_dim,
            d_state=d_state,
            expand=expand,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            mamba_from_trion=mamba_from_trion
        )

    def forward(self, content, style):
        return self.model(content, style)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train_model')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def shared_step(self, batch, step):
        content, style = batch['content'], batch['style']
        # print(content,style)

        batch_size = content.shape[0]

        Ics = []
        for i in range(0, batch_size):
            content_i = content[i:i + 1, :, :, :]
            style_i = style[i:i + 1, :, :, :]
            output_i = self.model(content_i, style_i)
            Ics.append(output_i)
        Ics = torch.cat(Ics, 0)

        output_Icc = []
        output_Iss = []
        '''
        Attention: mamba by trion can not inject various style information in a batch to A,D directly. 
        So we generate output stylized results one by one. This is similar to Gradient Accumulation.
        '''
        for i in range(0, batch_size):
            content_i = content[i:i + 1, :, :, :]
            Icc = self.model(content_i, content_i)
            output_Icc.append(Icc)

            style_i = style[i:i + 1, :, :, :]
            Iss = self.model(style_i, style_i)
            output_Iss.append(Iss)
        output_Icc = torch.cat(output_Icc, 0)
        output_Iss = torch.cat(output_Iss, 0)

        content_loss, style_loss, identity_loss1, identity_loss2 = self.loss(Ics, output_Icc, output_Iss, content,
                                                                             style)

        self.log(rf'id_loss1', identity_loss1.item(), prog_bar=step == 'train_model')
        self.log(rf'id_loss2', identity_loss2.item(), prog_bar=step == 'train_model')

        # Log metrics
        self.log(rf'loss_style', style_loss.item(), prog_bar=step == 'train_model')
        self.log(rf'loss_content', content_loss.item(), prog_bar=step == 'train_model')
        # print('loss_style', style_loss.item(),'------','loss_content', content_loss.item())

        # Return output only for validation step
        if step == 'val':
            self.validation_step_outputs.append({
                'loss': content_loss + style_loss + identity_loss1 + identity_loss2,
                'output': Ics,
                'content': content,
                'style': style
            })

            return {
                'loss': content_loss + style_loss + identity_loss1 + identity_loss2,
                'output': Ics,
            }

        return content_loss + style_loss + identity_loss1 + identity_loss2

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if self.global_step == 0:
            return

        with torch.no_grad():
            imgs = [x['output'] for x in outputs]
            imgs = [img for triple in imgs for img in triple]
            nrow = int(sqrt(len(imgs)))
            grid = make_grid(imgs, nrow=nrow, padding=0)
            logger = self.logger.experiment
            logger.add_image(rf'val_img', grid, global_step=self.global_step + 1)
            self.validation_step_outputs = []

    def loss(self, Ics, output_Icc, output_Iss, content, style):
        content_loss, style_loss, id1_loss, id2_loss = self.loss_func(Ics, output_Icc, output_Iss, content, style)

        return self.content_weight * content_loss, self.style_weight * style_loss, self.lambda1 * id1_loss, self.lambda2 * id2_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)

        def lr_lambda(iter):
            return 1 / (1 + 0.00002 * iter)

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
