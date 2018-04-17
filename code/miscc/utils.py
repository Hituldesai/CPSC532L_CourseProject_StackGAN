import os
import errno
import numpy as np

from copy import deepcopy
from miscc.config import cfg

from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils


#############################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def PIXEL_loss(real_imgs, fake_imgs):
    loss = nn.MSELoss()
    if cfg.CUDA:
        loss.cuda()
    fake = fake_imgs.detach()
    output = loss(real_imgs, fake)
    return output

def ACT_loss(fake_features, real_features):
    loss = nn.MSELoss()
    if cfg.CUDA:
        loss.cuda()
    fake_features = fake_features.detach()
    real_features = real_features.detach()
    output = loss(real_features, fake_features)
    return output

def TEXT_loss(gram, fake_features, real_features, weight):
    loss = nn.MSELoss()
    if cfg.CUDA:
        loss.cuda()
    gram_fake = gram(fake_features)
    gram_real = gram(real_features)
    gram_fake= gram_fake.detach()*weight
    gram_real = gram_real.detach()*weight
    output = loss(gram_fake, gram_real)
    return output

def compute_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels,
                               conditions, gpus, flag):
    if flag:
        criterion = nn.BCELoss()
        batch_size = real_imgs.size(0)
        cond = conditions.detach()
        fake = fake_imgs.detach()
        real_features = nn.parallel.data_parallel(netD, (real_imgs), gpus)
        fake_features = nn.parallel.data_parallel(netD, (fake), gpus)
        # real pairs
        inputs = (real_features, cond)
        real_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
        #errD_real = criterion(real_logits, real_labels)
        errD_real = real_logits
        # wrong pairs
        inputs = (real_features[:(batch_size-1)], cond[1:])
        wrong_logits = \
            nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
        #errD_wrong = criterion(wrong_logits, fake_labels[1:])
        errD_wrong = wrong_logits
        # fake pairs
        inputs = (fake_features, cond)
        fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
        #errD_fake = criterion(fake_logits, fake_labels)
        errD_fake = fake_logits

        if netD.get_uncond_logits is not None:
            real_logits = \
                nn.parallel.data_parallel(netD.get_uncond_logits,
                                          (real_features), gpus)
            fake_logits = \
                nn.parallel.data_parallel(netD.get_uncond_logits,
                                          (fake_features), gpus)
            #uncond_errD_real = criterion(real_logits, real_labels)
            #uncond_errD_fake = criterion(fake_logits, fake_labels)
            uncond_errD_real = real_logits
            uncond_errD_fake = fake_logits
            #
            errD = ((errD_real + uncond_errD_real) / 2. +
                    (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
            errD_real = (errD_real + uncond_errD_real) / 2.
            errD_fake = (errD_fake + uncond_errD_fake) / 2.
        else:
            errD = errD_real + (errD_fake + errD_wrong) * 0.5
        return errD, errD_real.data[0], errD_wrong.data[0], errD_fake.data[0]
    else:
        criterion = nn.BCELoss()
        batch_size = real_imgs.size(0)
        cond = conditions.detach()
        fake = fake_imgs.detach()
        real_features = netD(real_imgs)
        fake_features = netD(fake)
        # real pairs
        cond = cond.contiguous()
        real_logits = netD.get_cond_logits(real_features, cond)
        #errD_real = criterion(real_logits, real_labels)
        errD_real = real_logits
        # wrong pairs
        wrong_logits = netD.get_cond_logits(real_features[:(batch_size - 1)], cond[1:])
        #errD_wrong = criterion(wrong_logits, fake_labels[1:])
        errD_wrong = wrong_logits
        # fake pairs
        fake_logits = netD.get_cond_logits(fake_features, cond)
        #errD_fake = criterion(fake_logits, fake_labels)
        errD_fake = fake_logits

        if netD.get_uncond_logits is not None:
            real_logits = netD.get_uncond_logits(real_features)
            fake_logits = netD.get_uncond_logits(fake_features)
            #uncond_errD_real = criterion(real_logits, real_labels)
            #uncond_errD_fake = criterion(fake_logits, fake_labels)
            uncond_errD_real = real_logits
            uncond_errD_fake = fake_logits
            #
            errD = ((errD_real + uncond_errD_real) / 2. +
                    (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
            errD_real = (errD_real + uncond_errD_real) / 2.
            errD_fake = (errD_fake + uncond_errD_fake) / 2.
        else:
            errD = errD_real + (errD_fake + errD_wrong) * 0.5
        return errD, errD_real.data[0], errD_wrong.data[0], errD_fake.data[0]


def compute_generator_loss(netD, fake_imgs, real_labels, conditions, gpus, flag):
    if flag:
        criterion = nn.BCELoss()
        cond = conditions.detach()
        fake_features = nn.parallel.data_parallel(netD, (fake_imgs), gpus)
        # fake pairs
        inputs = (fake_features, cond)
        fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
        #errD_fake = criterion(fake_logits, real_labels)
        errD_fake = fake_logits
        if netD.get_uncond_logits is not None:
            fake_logits = \
                nn.parallel.data_parallel(netD.get_uncond_logits,
                                          (fake_features), gpus)
            #uncond_errD_fake = criterion(fake_logits, real_labels)
            uncond_errD_fake = fake_logits
            errD_fake += uncond_errD_fake
        return errD_fake
    else:
        criterion = nn.BCELoss()
        cond = conditions.detach()
        fake_features = netD(fake_imgs)
        cond = cond.contiguous()
        # fake pairs
        fake_logits = netD.get_cond_logits(fake_features, cond)
        #errD_fake = criterion(fake_logits, real_labels)
        errD_fake = fake_logits
        if netD.get_uncond_logits is not None:
            fake_logits = netD.get_uncond_logits(fake_features)
            #uncond_errD_fake = criterion(fake_logits, real_labels)
            uncond_errD_fake = fake_logits
            errD_fake += uncond_errD_fake
        return errD_fake


#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


#############################
def save_img_results(data_img, fake, epoch, image_dir):
    num = cfg.VIS_COUNT
    fake = fake[0:num]
    # data_img is changed to [0,1]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples.png' % image_dir,
            normalize=True)
        # fake.data is still [-1, 1]
        vutils.save_image(
            fake.data, '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)


def save_model(netG, netD, epoch, model_dir):
    torch.save(
        netG.state_dict(),
        '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        netD.state_dict(),
        '%s/netD_epoch_last.pth' % (model_dir))
    print('Save G/D models')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
