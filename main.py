import argparse
import os
import sys

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import utils as utils

from build import Model
from datasets.dataset import MapDataset

FLAGS = None


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device = }")

    all_data = "root/to/dataset/"
    _test_mprage = os.path.join(all_data, "name_test_src1.npy")
    _test_ct = os.path.join(all_data, "name_test_src2.npy")
    test_img = MapDataset([_test_mprage, _test_ct], train=False)
    test_dataloader = DataLoader(test_img, batch_size=FLAGS.batch_size_test,
                                shuffle=False, num_workers=os.cpu_count(),
                                drop_last=True)

    model = Model(FLAGS.model, device, None, test_data_loader=test_dataloader, FLAGS=FLAGS)
    print('Loading Model')
    model.load_from(FLAGS.out_dir_train, name='gen')
    print('Evaluating Model')
    model.eval(batch_size=FLAGS.batch_size_test, out_dir=FLAGS.out_dir_test, save_figs=FLAGS.save_figs)
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multimodal Medical Image Fusion Using an Unsupervised Deep '
                                                 'Generative Adversarial Network.')
    parser.add_argument('--model', type=str, default='MedFusionGAN', help='Fusion Model')
    parser.add_argument('--cuda', type=utils.boolean_string, default=True, help='enable CUDA.')
    parser.add_argument('--train', type=utils.boolean_string, default=False, help='train mode or eval mode.')
    parser.add_argument('--out_dir_train', type=str, default='./results/train/out1', help='Directory for training output.')
    parser.add_argument('--out_dir_test', type=str, default='./results/test/out1', help='Directory for testing output.')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
    parser.add_argument('--batch_size_train', type=int, default=8, help='size of batches in training')
    parser.add_argument('--batch_size_test', type=int, default=1, help='size of batches in inference')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--optimizer_betas', type=tuple, default=(0.5, 0.9999), help='Adam optimizer Betas')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--lambda_l1', type=float, default=1.0, help='Identity loss weight')
    parser.add_argument('--lambda_perce', type=float, default=0.2, help='Perceptual loss weight')
    parser.add_argument('--lambda_perceMRI', type=float, default=0.1, help='Perceptual loss weight of MRI')
    parser.add_argument('--lambda_ssimCT', type=float, default=1, help='SSIM loss weight for CT')
    parser.add_argument('--lambda_ssim_mri', type=float, default=0.60, help='SSIM loss weight for CT')
    parser.add_argument('--lambda_qilv', type=float, default=1.0, help='QILV loss weight for MRI')
    parser.add_argument('--lambda_tvCT', type=float, default=1.0, help='Total Variation loss weight for CT')
    parser.add_argument('--lambda_tvMR', type=float, default=1.0, help='Total Variation loss weight for MR')
    parser.add_argument('--lambda_adv', type=float, default=1.0, help='Adversarial loss weight')
    parser.add_argument('--lambda_vif', type=float, default=1.0, help='VIF loss weight')

    FLAGS = parser.parse_args()
    FLAGS.cuda = FLAGS.cuda and torch.cuda.is_available()

    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)
        if FLAGS.cuda:
            torch.cuda.manual_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    cudnn.benchmark = True

    if FLAGS.train:
        utils.clear_folder(FLAGS.out_dir_train)


        log_file = os.path.join(FLAGS.out_dir_train, 'log.txt')
        print("Logging to {}\n".format(log_file))
        sys.stdout = utils.StdOut(log_file)
    else:
        utils.clear_folder(FLAGS.out_dir_test)
        log_file = os.path.join(FLAGS.out_dir_test, 'log.txt')
        print("Logging to {}\n".format(log_file))
        sys.stdout = utils.StdOut(log_file)

    print("PyTorch version: {}".format(torch.__version__))
    print("CUDA version: {}\n".format(torch.version.cuda))
    # print("Similar to out3 with my perceptual loss function\nSelf Att. Unet\nNN loss and LeakyReLU activations.")
    print(" " * 9 + "Args" + " " * 9 + "|    " + "Type" + \
          "    |    " + "Value")
    print("-" * 50)
    for arg in vars(FLAGS):
        arg_str = str(arg)
        var_str = str(getattr(FLAGS, arg))
        type_str = str(type(getattr(FLAGS, arg)).__name__)
        print("  " + arg_str + " " * (20-len(arg_str)) + "|" + \
              "  " + type_str + " " * (10-len(type_str)) + "|" + \
              "  " + var_str)

    main()
