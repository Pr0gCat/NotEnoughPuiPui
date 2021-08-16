#!/usr/bin/env python
# coding: utf-8

# # 用DCGAN方法GAN出更多天竺鼠車車圖片
#
# **Won't work, model keep collapsing**
#

from numpy.core.shape_base import block
import torch
import torch.nn as nn
import torch.optim as optimizers
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import random
import numpy as np
from torchsummary import summary

import matplotlib.pyplot as plt

from dcgan import NEPPDiscriminator, NEPPGenerator

manualSeed = 666
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ## 定義生成網路
g_net = NEPPGenerator().cuda()
g_net.apply(weights_init)
print(g_net)


# ## 定義評論網路
d_net = NEPPDiscriminator().cuda()
d_net.apply(weights_init)
summary(d_net, (3,64,64))


# ## 準備訓練資料
puipui_imgs = ImageFolder(root='puipui', transform=transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.RandomHorizontalFlip(0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

print('number of images:', len(puipui_imgs.imgs))
# puipui_imgs[0][0]


# ## 超參數
EPOCHS = 20000
BATCH_SIZE = 50
LR_G = 0.0001
LR_D = 0.0001


# ## 訓練
dataloader = DataLoader(puipui_imgs, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

loss_fn = nn.BCELoss()

g_optim = optimizers.Adam(g_net.parameters(), lr=LR_G)
d_optim = optimizers.Adam(d_net.parameters(), lr=LR_D)

generated = []
fixed_noise = torch.randn(32, 100, 1, 1).cuda()

import random

#display(status)
block_d = False
for epoch in range(1, EPOCHS+1):
    DRL_loss = 0
    DFL_loss = 0
    total_d_loss = 0
    total_g_loss = 0
    tp = 0
    tn = 0
    out_real = []
    out_fake = []
    success_fake = []
    t_success_fake = 0
    t_total = 0
    for i, batch in enumerate(dataloader):
        # 用真的圖更新評論網路
        d_net.zero_grad()
        x = batch[0].cuda()
        x_size = x.size(0)
        d_pred = d_net(x).view(-1)
        out_real += d_pred.detach().tolist()
        is_real = torch.ones((x_size,), dtype=torch.float).cuda()
        real_loss = loss_fn(d_pred, is_real)
        real_loss.backward()
        d_optim.step()

        DRL_loss += real_loss.mean().item()
        tp += (d_pred.detach() >= 0.5).sum().item()
        t_total += x_size

        # 用假的圖做更新評論網路
        random_noise = torch.randn(x_size, 100, 1, 1).cuda()
        d_net.zero_grad()
        is_fake = torch.zeros((x_size,), dtype=torch.float).cuda()
        d_pred = d_net(g_net(random_noise).detach()).view(-1)
        out_fake += d_pred.detach().tolist()
        fake_loss = loss_fn(d_pred, is_fake)
        # block d when d is great better than g
        if not block_d:
            fake_loss.backward()
            d_optim.step()

        DFL_loss += fake_loss.mean().item()
        tn += (d_pred.detach() < 0.5).sum().item()

        d_loss = real_loss + fake_loss
        total_d_loss += d_loss.item()

        # 更新生成網路
        g_net.zero_grad()
        # d_net.eval() # attempt 1: only update g_net
        d_pred = d_net(g_net(random_noise)).view(-1)
        is_real = torch.ones((x_size,), dtype=torch.float).cuda()
        g_loss = loss_fn(d_pred, is_real)
        g_loss.backward()
        g_optim.step()
        total_g_loss += g_loss.item()
        success_fake += d_pred.detach().tolist()
        t_success_fake += (d_pred.detach() >= 0.5).sum().item()
        # d_net.train()

    if (tn / t_total * 100) - (t_success_fake/t_total * 100) > 50:
        print('Block D')
        block_d = True
    else:
        block_d = False

    print(f'EPOCH {epoch} \n tp acc: {round(tp / t_total * 100, 2)}% | tn acc: {round(tn / t_total * 100, 2)}% \n DR mean/std: {round(np.mean(out_real), 6)}/{round(np.std(out_real), 6)} | DF mean/std: {round(np.mean(out_fake), 6)}/{round(np.std(out_fake), 6)} \n G acc: {t_success_fake/t_total * 100}% | Gout mean/std: {round(np.mean(success_fake), 6)}/{round(np.std(success_fake), 6)} \n DR loss: {DRL_loss} | DF loss: {DFL_loss} \n DT loss: {total_d_loss} | GT loss: {total_g_loss}\n\n', end='\r', flush=True)
    if epoch % 100 == 0:
        print('Checkpoint')
        with torch.no_grad():
            generated_puipui = g_net(fixed_noise).detach().cpu()
            fig = plt.figure(figsize=(8,8))
            plt.axis("off")
            img = make_grid(generated_puipui, padding=2, normalize=True)
            plt.imsave(f'dcgan_results/{epoch}.jpg', np.transpose(img,(1,2,0)).numpy())

        torch.save(g_net.state_dict(), 'dcgan_models/g_net.pth')
        torch.save(d_net.state_dict(), 'dcgan_models/d_net.pth')
