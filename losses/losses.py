import torch
import torchvision
import torch.nn.functional as F
from math import exp
from torch import nn
import numpy as np


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, device=None):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        vgg = torchvision.models.vgg16(pretrained=True).to(device)
        vgg.eval()
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        input, target = input * 0.5 + 0.5, target * 0.5 + 0.5
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            # target = torch.cat([target[:,0,:,:].unsqueeze(1), target[:,1,:,:].unsqueeze(1), target[:,1,:,:].unsqueeze(1)], dim = 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        # if self.resize:
        #     input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        #     target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class vgg_perceptual_loss(torch.nn.Module):
    def __init__(self, device=None):
        super(vgg_perceptual_loss, self).__init__()
        blocks = []
        vgg = torchvision.models.vgg16(pretrained=True).to(device)

        vgg.eval()
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        # self.blocks = self.replace_active(self.blocks)

    def replace_active(self, model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, child_name, nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                self.replace_active(child)
        return model

    def forward(self, input, target):
        input = input * 0.5 + 0.5
        target = target * 0.5 + 0.5
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.mse_loss(x, y)

        return loss


class SSIM_Loss:
    def __init__(self, window_size=11, window=None, size_average=True, full=False, val_range=None):
        self.window_size = window_size
        self.window = window
        self.size_average = size_average
        self.full = full
        self.val_range = val_range

    @staticmethod
    def gaussian(window_size, sigma):
        # gauss = torch.tensor([exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
        x_cord = torch.arange(window_size)
        x_grid = x_cord.repeat(window_size).view(window_size, window_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim = -1)
        gauss = (1./(2.*3.14*sigma ** 2)) *\
                  torch.exp(
                      -torch.sum((xy_grid - window_size // 2)**2., dim=-1) /\
                      (2*sigma**2)
                  ).unsqueeze(0).unsqueeze(0)

        return gauss/gauss.sum()


    def create_window(self, window_size, channel=1):
        # _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        # _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = self.gaussian(window_size, 1.5).repeat(channel, 1, 1, 1)
        return window


    def ssim_loss(self, img1, img2):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if self.val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = self.val_range

        padd = 0
        # img1 = torch.cat([img1, img1], dim = 1)
        (_, channel, height, width) = img1.size()
        if self.window is None:
            real_size = min(self.window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(img1.device)
            # print(window.shape)
            # exit()
        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = v1 / v2  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
        # print(ssim_map.shape)
        # exit()
        if self.size_average:
            cs = cs.mean()
            ret = ssim_map.mean()
        else:
            cs = cs.mean(1).mean(1).mean(1)
            ret = ssim_map.mean(1).mean(1).mean(1)
        if ret > 1:
            return ret - 1
        else:
            return 1 - ret
        # if self.full:
        #     return 1-ret + 1-cs
        # else:
        #     # return 1 - 0.5 * (ret + 1)
        #     return 1 - ret



class QILV_Loss(nn.Module):
    """
    Quality Index based on Local Variance
    """
    def __init__(self, window_size=21, window=None):
        super(QILV_Loss, self).__init__()
        self.window_size = window_size
        self.window = window

    @staticmethod
    def gaussian(window_size, sigma):
        # gauss = torch.tensor([exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
        x_cord = torch.arange(window_size)
        x_grid = x_cord.repeat(window_size).view(window_size, window_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim = -1)
        gauss = (1./(2.*3.14*sigma ** 2)) *\
                  torch.exp(
                      -torch.sum((xy_grid - window_size // 2)**2., dim=-1) /\
                      (2*sigma**2)
                  ).unsqueeze(0).unsqueeze(0)

        return gauss/gauss.sum()


    def create_window(self, window_size, channel=1):
        # _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        # _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = self.gaussian(window_size, 1.5).repeat(channel, 1, 1, 1)

        return window

    def forward(self, img1, img2):
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        padd = 0
        # img1 = torch.cat([img1, img1], dim = 1)
        (_, channel, height, width) = img1.size()
        if self.window is None:
            real_size = min(self.window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        var_i1 = F.conv2d(img1.pow(2), window, padding=padd, groups=channel) - mu1.pow(2)
        var_i2 = F.conv2d(img2.pow(2), window, padding=padd, groups=channel) - mu2.pow(2)

        mu_v1 = torch.mean(var_i1)
        mu_v2 = torch.mean(var_i2)

        sigma_v1 = torch.std(var_i1 - var_i1.mean())
        sigma_v2 = torch.std(var_i2 - var_i2.mean())
        sigma_v12 = torch.std((var_i2 - var_i2.mean()) * (var_i1 - var_i1.mean()))


        # qilv= (2 * mu_v1 * mu_v2 + C1) / (mu_v1.pow(2) + mu_v2.pow(2) + C1) * \
        #       (2 * sigma_v1 * sigma_v2 + C2) / (sigma_v1.pow(2) + sigma_v2.pow(2) + C2) *\
        #       (sigma_v12 + C2 / 2) / (sigma_v1 * sigma_v2 + C2/2)
        qilv = (2 * mu_v1 * mu_v2) / (mu_v1.pow(2) + mu_v2.pow(2)) * 2 / (sigma_v1 + sigma_v2) * sigma_v12
        if qilv > 1:
            return qilv - 1
        else:
            return 1 - qilv


#################### FID #########################

class FID(nn.Module):
    r"""
    Inception Score PyTorch port from the official Tensorflow implementation.
    Tensorflow Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
    Paper: https://arxiv.org/pdf/1706.08500.pdf
    """

    def __init__(self, device, dims = 2048):
        super(FID, self).__init__()

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.add_module('inception', InceptionV3([block_idx]))
        self.inception.eval()

    def __call__(self, generated, real):
        if generated.shape[1] != 3:
            generated = generated.repeat(1, 3, 1, 1)
            real = real.repeat(1, 3, 1, 1)
        activations_r = self.inception(real)[0].squeeze().double()
        activations_g = self.inception(generated)[0].squeeze().double()

        num_examples_r = activations_r.shape[0]
        num_examples_g = activations_g.shape[0]

        m_r = torch.mean(activations_r, dim = 0)
        m_g = torch.mean(activations_g, dim = 0)

        # centered_r = torch.unsqueeze(activations_r - m_r, 0)
        # centered_r = activations_r - m_r
        # sigma_r = centered_r.t().matmul(centered_r) / (num_examples_r - 1)
        # # sigma_r = torch.matmul(torch)
        #
        # # centered_g = torch.unsqueeze(activations_g - m_g, 0)
        # centered_g = activations_g - m_g
        # sigma_g = centered_g.t().matmul(centered_g) / (num_examples_g - 1)
        #
        # sqrt_trace_component = self._trace_sqrt_product(sigma_r, sigma_g)
        # trace = torch.trace(sigma_r + sigma_g) - 2.0 * sqrt_trace_component

        mean = torch.sum((m_r - m_g) ** 2)
        # fid = trace + mean

        # return fid.item()
        return mean

    def _trace_sqrt_product(self, sigma_r, sigma_g):
        '''
        Find the trace of the positive sqrt of product of covariance matrices.
        '_symmetric_matrix_square_root' only works for symmetric matrices, so we
        cannot just take _symmetric_matrix_square_root(sigma_r * sigma_g).
        ('sigma_r' and 'sigma_g' are symmetric, but their product is not necessarily).
        Let sigma_r = A A so A = sqrt(sigma_r), and sigma_g = B B.
        We want to find trace(sqrt(sigma_r sigma_g)) = trace(sqrt(A A B B))
        Note the following properties:
        (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
            => eigenvalues(A A B B) = eigenvalues (A B B A)
        (ii) eigenvalues(sqrt(M)) = sqrt(eigenvalues(M))
            => eigenvalues(sqrt(sigma_r sigma_g)) = sqrt(eigenvalues(A B B A))
        (iii) forall M: trace(M) = sum(eigenvalues(M))
            => trace(sqrt(sigma_r sigma_g)) = sum(eigenvalues(sqrt(sigma_r sigma_g)))
                                            = sum(sqrt(eigenvalues(A A B B)))
                                            = sum(sqrt(eigenvalues(A B B A)))
                                            = sum(eigenvalues(sqrt(A B B A)))
                                            = trace(sqrt(A B B A))
                                            = trace(sqrt(A sigma_g A))
        A = sqrt(sigma_r). Both sigma_r and A sigma_g A are symmetric, so we **can**
        use the _symmetric_matrix_square_root function to find the roots of these
        matrices.
        Args:
            sigma_r: a square, symmetric, real, positive semi-definite covariance matrix
            sigma_g: same as sigma_r
        Returns:
            The trace of the positive square root of sigma_r*sigma_g
        '''
        sqrt_sigma_r = self._symmetric_matrix_square_root(sigma_r)
        sqrt_a_sigmav_a = sqrt_sigma_r.matmul(sigma_g.matmul(sqrt_sigma_r))
        trace = torch.trace(self._symmetric_matrix_square_root(sqrt_a_sigmav_a))

        return trace

    def _symmetric_matrix_square_root(self, mat, eps = 1e-10):
        '''
        Compute square root of a symmetric matrix.
        Note:
            Let A = R^2 be a symmetric matrix, we want to find R
            SVD decomposition for A:
                U.S.V^T = SVD(A)

            Let     D = S^1/2
            Then    (U.D.V^T)^2 = (U.D.V^T).(U.D.V^T) = U.D^2.V^T = U.S.V^T = A = M^2
                    => M = U.D.V^T
                    => M = U.S^1/2.V^T
        Args:
            mat: Matrix to take the square root of.
            eps: Small epsilon such that any element less than eps will not be square
            rooted to guard against numerical instability.
        Returns:
            Matrix square root of mat.
        '''

        # singular value dicomposition
        u, s, v = torch.svd(mat)
        # sqrt is unstable around 0, just use 0 in such case
        si = torch.where(s < eps, s, torch.sqrt(s))
        # U.S^1/2.V^T
        res = u.matmul(torch.diag(si)).matmul(v.t())

        return res


class InceptionV3(nn.Module):
    '''https://github.com/mseitzer/pytorch-fid/blob/master/inception.py'''

    # Pretrained InceptionV3 network returning feature maps

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks = [DEFAULT_BLOCK_INDEX],
                 resize_input = True,
                 normalize_input = True,
                 requires_grad = False):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()
        DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        inception = torchvision.models.inception_v3(pretrained = True).to(DEVICE)
        inception.eval()
        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size = 3, stride = 2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size = (1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            # x = F.upsample(x, size = (299, 299), mode = 'bilinear')
            x = F.interpolate(x, size = (299, 299), mode = 'bilinear',align_corners=True)

        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
#
# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#     return gauss / gauss.sum()
#
#
# def create_window(window_size, channel=1):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
#     return window
#
#
# def _mef_ssim(imgSeq, refImg, window, window_size):
#     (_, imgSeq_channel, _, _) = imgSeq.size()
#     (_, refImg_channel, _, _) = refImg.size()
#     C2 = (0.03 * 255) ** 2
#     sWindow = torch.ones((imgSeq_channel, 1, window_size, window_size)).to(imgSeq.device) / window_size ** 2
#     mu_x = F.conv2d(imgSeq, sWindow, padding=window_size // 2, groups=imgSeq_channel)
#
#     mfilter = torch.ones((imgSeq_channel, 1, window_size, window_size)).to(imgSeq.device)
#     x_hat = imgSeq - mu_x
#     x_hat_norm = torch.sqrt(
#         F.conv2d(torch.pow(x_hat, 2), mfilter, padding=window_size // 2, groups=imgSeq_channel)) + 0.001
#     c_hat = torch.unsqueeze(torch.max(x_hat_norm, dim=1)[0], 1)
#
#     mfilter2 = torch.ones((1, 1, window_size, window_size)).to(imgSeq.device)
#     R = (torch.sqrt(F.conv2d(torch.pow(torch.sum(x_hat, 1, keepdim=True), 2), mfilter2, padding=window_size // 2,
#                              groups=1)) + np.spacing(1) + np.spacing(1)) \
#         / (torch.sum(x_hat_norm, 1, keepdim=True) + np.spacing(1))
#
#     R[R > 1] = 1 - np.spacing(1)
#     R[R < 0] = 0 + np.spacing(1)
#
#     p = torch.tan(R * np.pi / 2)
#     p[p > 10] = 10
#
#     s = x_hat / x_hat_norm
#
#     s_hat_one = torch.sum((torch.pow(x_hat_norm, p) + np.spacing(1)) * s, 1, keepdim=True) / torch.sum(
#         (torch.pow(x_hat_norm, p) + np.spacing(1)), 1, keepdim=True)
#     s_hat_two = s_hat_one / torch.sqrt(
#         F.conv2d(torch.pow(s_hat_one, 2), mfilter2, padding=window_size // 2, groups=refImg_channel))
#
#     x_hat_two = c_hat * s_hat_two
#
#     mu_x_hat_two = F.conv2d(x_hat_two, window, padding=window_size // 2, groups=refImg_channel)
#     mu_y = F.conv2d(refImg, window, padding=window_size // 2, groups=refImg_channel)
#
#     mu_x_hat_two_sq = torch.pow(mu_x_hat_two, 2)
#     mu_y_sq = torch.pow(mu_y, 2)
#     mu_x_hat_two_mu_y = mu_x_hat_two * mu_y
#     sigma_x_hat_two_sq = F.conv2d(x_hat_two * x_hat_two, window, padding=window_size // 2,
#                                   groups=refImg_channel) - mu_x_hat_two_sq
#     sigma_y_sq = F.conv2d(refImg * refImg, window, padding=window_size // 2, groups=refImg_channel) - mu_y_sq
#     sigmaxy = F.conv2d(x_hat_two * refImg, window, padding=window_size // 2, groups=refImg_channel) - mu_x_hat_two_mu_y
#
#     mef_ssim_map = (2 * sigmaxy + C2) / (sigma_x_hat_two_sq + sigma_y_sq + C2)
#
#     return mef_ssim_map.mean()
#
#
# class MEF_SSIM(torch.nn.Module):
#     def __init__(self, window_size=11):
#         super(MEF_SSIM, self).__init__()
#         self.window_size = window_size
#         self.channel = 1
#         self.window = create_window(window_size, self.channel)
#
#     def forward(self, img_seq, refImg):
#         if img_seq.is_cuda:
#             self.window = self.window.cuda(img_seq.get_device())
#         self.window = self.window.type_as(img_seq)
#
#         return _mef_ssim(img_seq, refImg, self.window, self.window_size)
#
#
# def mef_ssim(img_seq, refImg, window_size=11):
#     (_, channel, _, _) = refImg.size()
#     window = create_window(window_size, channel)
#
#     if img_seq.is_cuda:
#         window = window.cuda(img_seq.get_device())
#     window = window.type_as(img_seq)
#
#     return _mef_ssim(img_seq, refImg, window, window_size)


class MEF_SSIM():
    def __init__(self, device, window_size=11):
        # super(MEF_SSIM, self).__init__()
        self.window_size = window_size
        self.window = self.create_window(1).to(device, dtype=torch.float)

        self.eps = torch.finfo(torch.float64).eps

    def gaussian(self, sigma):
        gauss = torch.Tensor([exp(-(x - self.window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(self.window_size)])
        return gauss / gauss.sum()

    def create_window(self, channel):
        _1D_window = self.gaussian(1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, self.window_size, self.window_size).contiguous()
        return window

    def _mef_ssim(self, imgSeq, refImg):


        (_, imgSeq_channel, _, _) = imgSeq.size()
        (_, refImg_channel, _, _) = refImg.size()
        # window = self.create_window(imgSeq_channel).to(imgSeq.device)

        C2 = (0.03 * 255) ** 2
        sWindow = torch.ones((imgSeq_channel, 1, self.window_size, self.window_size)).to(imgSeq.device) / self.window_size ** 2

        mu_x = F.conv2d(imgSeq, sWindow, padding = self.window_size // 2, groups = imgSeq_channel)

        mfilter = torch.ones((imgSeq_channel, 1, self.window_size, self.window_size)).to(imgSeq.device)


        x_hat = imgSeq - mu_x
        x_hat_norm = torch.sqrt(
            F.conv2d(torch.pow(x_hat, 2), mfilter, padding = self.window_size // 2, groups = imgSeq_channel)) + 0.001

        # print()
        c_hat = torch.unsqueeze(torch.max(x_hat_norm, dim = 1)[0], 1)

        mfilter2 = torch.ones((1, 1, self.window_size, self.window_size)).to(imgSeq.device)
        R = (torch.sqrt(
            F.conv2d(torch.pow(torch.sum(x_hat, 1, keepdim = True), 2), mfilter2, padding = self.window_size // 2,
                     groups = 1)) + self.eps + self.eps) \
            / (torch.sum(x_hat_norm, 1, keepdim = True) + self.eps)

        R[R > 1] = 1 - self.eps
        R[R < 0] = 0 + self.eps

        p = torch.tan(R * 3.14 / 2)
        p[p > 10] = 10

        s = x_hat / x_hat_norm

        s_hat_one = torch.sum((torch.pow(x_hat_norm, p) + self.eps) * s, 1, keepdim = True) / torch.sum(
            (torch.pow(x_hat_norm, p) + self.eps), 1, keepdim = True)

        s_hat_two = s_hat_one / torch.sqrt(
            F.conv2d(torch.pow(s_hat_one, 2), mfilter2, padding = self.window_size // 2, groups = refImg_channel))

        x_hat_two = c_hat * s_hat_two

        mu_x_hat_two = F.conv2d(x_hat_two, self.window, padding = self.window_size // 2, groups = refImg_channel)
        mu_y = F.conv2d(refImg, self.window, padding = self.window_size // 2, groups = refImg_channel)

        mu_x_hat_two_sq = torch.pow(mu_x_hat_two, 2)
        mu_y_sq = torch.pow(mu_y, 2)
        mu_x_hat_two_mu_y = mu_x_hat_two * mu_y
        sigma_x_hat_two_sq = F.conv2d(x_hat_two * x_hat_two, self.window, padding = self.window_size // 2,
                                      groups = refImg_channel) - mu_x_hat_two_sq
        sigma_y_sq = F.conv2d(refImg * refImg, self.window, padding = self.window_size // 2, groups = refImg_channel) - mu_y_sq
        sigmaxy = F.conv2d(x_hat_two * refImg, self.window, padding = self.window_size // 2,
                           groups = refImg_channel) - mu_x_hat_two_mu_y

        mef_ssim_map = (2 * sigmaxy + C2) / (sigma_x_hat_two_sq + sigma_y_sq + C2)

        return mef_ssim_map.mean()


    def loss(self, img_seq, refImg):
        # if img_seq.is_cuda:
            # self.window = self.window.cuda(img_seq.get_device())
        # self.window = self.window.type_as(img_seq)

        return 1 - self._mef_ssim(img_seq, refImg)

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#
# x = torch.randn((8, 1, 32, 32)).to(device, dtype=torch.float)
# y = torch.randn((8, 1, 32, 32)).to(device, dtype=torch.float)
# loss = MEF_SSIM(device)
# print(loss(x, y))

class MEF_SSIM_Loss:
    def __init__(self, window_size=11, window=None, size_average=True, full=False, val_range=None):
        self.window_size = window_size
        self.window = window
        self.size_average = size_average
        self.full = full
        self.val_range = val_range

    @staticmethod
    def gaussian(window_size, sigma):
        # gauss = torch.tensor([exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
        x_cord = torch.arange(window_size)
        x_grid = x_cord.repeat(window_size).view(window_size, window_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim = -1)
        gauss = (1./(2.*3.14*sigma ** 2)) *\
                  torch.exp(
                      -torch.sum((xy_grid - window_size // 2)**2., dim=-1) /\
                      (2*sigma**2)
                  ).unsqueeze(0).unsqueeze(0)

        return gauss/gauss.sum()


    def create_window(self, window_size, channel=1):
        # _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        # _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = self.gaussian(window_size, 1.5).repeat(channel, 1, 1, 1)
        return window


    def mef_ssim_loss(self, img1, img2):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if self.val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = self.val_range

        padd = 0
        # img1 = torch.cat([img1, img1], dim = 1)
        (_, channel, height, width) = img1.size()
        if self.window is None:
            real_size = min(self.window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(img1.device)
            # print(window.shape)
            # exit()
        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1.pow(2)
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2.pow(2)
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - (mu1 * mu2)

        # C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2


        return 1 - torch.mean((2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))

class TV_loss(nn.Module):
    def __init__(self, REGULARIZATION=1e-3):
        super(TV_loss, self).__init__()
        self.regularization = REGULARIZATION
        self.l1 = torch.nn.L1Loss()
    def forward(self, x, y):
        reg_loss_x = self.regularization* (
                torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) +
                torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
        reg_loss_y = self.regularization * (
                torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))
        return self.l1(reg_loss_x, reg_loss_y)

class TV_loss2(nn.Module):
    def __init__(self, REGULARIZATION=1e-3):
        super(TV_loss2, self).__init__()
        self.regularization = REGULARIZATION
        self.l1 = torch.nn.L1Loss()
    def forward(self, x, y):
        img = x - y
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return self.regularization * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

        # return self.l1(reg_loss_x, reg_loss_y)
class VIF_loss(nn.Module):
    def __init__(self, window_size=55, window=None, size_average=True, full=False, val_range=None):
        super(VIF_loss, self).__init__()
        self.window_size = window_size
        self.window = window
        self.size_average = size_average
        self.full = full
        self.val_range = val_range

    @staticmethod
    def gaussian(window_size, sigma):
        # gauss = torch.tensor([exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
        x_cord = torch.arange(window_size)
        x_grid = x_cord.repeat(window_size).view(window_size, window_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim = -1)
        gauss = (1./(2.*3.14*sigma ** 2)) *\
                  torch.exp(
                      -torch.sum((xy_grid - window_size // 2)**2., dim=-1) /\
                      (2*sigma**2)
                  ).unsqueeze(0).unsqueeze(0)

        return gauss/gauss.sum()


    def create_window(self, window_size, sigma=1.5, channel=1):
        # _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        # _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = self.gaussian(window_size, sigma).repeat(channel, 1, 1, 1)
        return window


    def ComVidVindG(self, img1, img2):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        padd = 0
        # img1 = torch.cat([img1, img1], dim = 1)
        (_, channel, height, width) = img1.size()
        G, Num, Den = [], [], []
        sigma_nsq = 0.05
        for scale in range(1, 5):
            window_size = self.window_size
        # if self.window is None:
            window = self.create_window(window_size, sigma = window_size / 5 , channel=channel).to(img1.device)
            mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
            mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + 1e-10)

            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < 1e-10] = 0
            sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
            sigma1_sq[sigma1_sq < 1e-10] = 0

            g[sigma2_sq < 1e-10] = 0
            sv_sq[sigma2_sq < 1e-10] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= 1e-10] = 1e-10
            G.append(g)
            VID = torch.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq))
            VIND = torch.log10(1 + sigma1_sq / sigma_nsq)

            Num.append(VID)
            Den.append(VIND)

        return G, Num, Den

    def forward(self, img1, img2, imgF):
        C = 1e-7
        T1N, T1D, T1G = self.ComVidVindG(img1, imgF)
        T2N, T2D, T2G = self.ComVidVindG(img2, imgF)
        F = 0.0
        for i in range(3):
            M_Z1 = T1N[i]
            M_Z2 = T2N[i]
            M_M1 = T1D[i]
            M_M2 = T2D[i]
            M_G1 = T1G[i]
            M_G2 = T2G[i]
            L = M_G1 < M_G2
            M_G = M_G2
            M_G[L] = M_G1[L]
            M_Z12 = M_Z2
            M_Z12[L] = M_Z1[L]
            M_M12 = M_M2
            M_M12[L] = M_M1[L]

            VID = torch.sum(M_Z12 + C)
            VIND = torch.sum(M_M12 + C)
            # print(f"{VID=}")
            # print(f"{VIND=}")
            F += VID / VIND
        return 1- F * 0.01



class GradientLoss(nn.Module):
    def __init__(self, Device):
        super(GradientLoss, self).__init__()
        window_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).to(Device)
        self.window_x = window_x.view((1, 1, 3, 3))

        window_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(Device)
        self.window_y = window_y.view((1, 1, 3, 3))
    def forward(self, x, y):
        G_x = torch.sqrt(torch.pow(F.conv2d(x, self.window_x), 2) + torch.pow(F.conv2d(x, self.window_y), 2))
        G_y = torch.sqrt(torch.pow(F.conv2d(y, self.window_x), 2) + torch.pow(F.conv2d(y, self.window_y), 2))

        return F.l1_loss(G_x, G_y)


# x = torch.randn((1, 128, 256, 256)).cuda()
# y = torch.randn((1, 128, 256, 256)).cuda()
# z = torch.randn((1, 128, 256, 256)).cuda()
# loss = VIF_loss()
# print(loss(x, y, z))