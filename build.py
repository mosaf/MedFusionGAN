import os

import numpy as np
from tqdm import tqdm

from networks.Generator import Gen_unet02 as G
from networks.Discriminator import Discriminator as D
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
from losses import losses



def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Model(object):
    def __init__(self,
                 name,
                 device,
                 data_loader,
                 test_data_loader,
                 FLAGS):
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.test_data_loader = test_data_loader
        self.lambda_recon = FLAGS.lambda_reconstruction
        self.flags = FLAGS
        assert self.name == 'diffusion'

        self.G = G(in_channel=2 * self.flags.channels, out_channel=self.flags.channels)
        self.G.apply(_weights_init)
        self.G.to(self.device)

        self.D = D(in_channels=self.flags.channels)
        self.D.apply(_weights_init)
        self.D.to(self.device)

        self.loss_l1 = torch.nn.L1Loss()
        self.loss_perc = losses.VGGPerceptualLoss(device=self.device)
        self.loss_ssim = losses.SSIM_Loss().ssim_loss
        self.loss_qilv = losses.QILV_Loss()
        self.tv_loss = losses.TV_loss2(REGULARIZATION=1)

    @property
    def gen(self):
        return self.G

    @property
    def disc(self):
        return self.D

    def create_optim(self, lr, alpha=.9, beta=0.999):
        self.optim_G = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(alpha, beta))
        self.optim_D = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(alpha, beta))



    def eval(self,
             batch_size=None,
             out_dir=None,
             save_figs=True):
        self.gen.eval()

        if batch_size is None:
            batch_size = self.test_data_loader.batch_size

        Fused = torch.zeros((2276, 256, 256))  # (slice_number, dim1, dim2)
        with torch.inference_mode():
            for batch_idx, data_ in enumerate(tqdm(self.test_data_loader)):
                mri = data_['mri'].to(self.device, dtype=torch.float)  # mprage MRI
                ct = data_['ct'].to(self.device, dtype=torch.float)  # flair MRI

                fused = self.gen(mri, ct)
                Fused[batch_idx] = fused
                if save_figs:
                    viz_sample = torch.cat((mri, ct, fused), 0)
                    vutils.save_image(viz_sample,
                                      os.path.join(out_dir, 'sample_{}.png'.format(batch_idx)),
                                      nrow=batch_size,
                                      normalize=False)



            np.save(os.path.join(out_dir, 'fused.npy'), Fused.cpu().detach())
            torch.save(F, os.path.join(out_dir, 'Fused.pt'))


            print("Done evaluation.")



    def load_from(self,
                  path='',
                  name=None,
                  verbose=True):
        if name is None:
            name = self.name
        if verbose:
            print('\nLoading models from {}.pth.tar and such ...'.format(name))
        ckpt_G = torch.load(os.path.join(path, '{}.pth.tar'.format(name)))
        if isinstance(ckpt_G, dict) and 'state_dict' in ckpt_G:
            self.gen.load_state_dict(ckpt_G['state_dict'], strict=True)
        else:
            self.gen.load_state_dict(ckpt_G, strict=True)
