# # 用WGAN方法GAN出更多天竺鼠車車圖片
#
# 產生圖片大小: 64x64
#
from os import path
import torch
import torch.nn as nn
import torch.optim as optimizers
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import random
import numpy as np
import matplotlib.pyplot as plt

from wgan import NEPPDiscriminator, NEPPGenerator

G_NET_PATH = 'wgan_models/g_net.pth'
D_NET_PATH = 'wgan_models/d_net.pth'

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

## 定義生成網路
g_net = NEPPGenerator().cuda()
if path.exists(G_NET_PATH):
    print(f'Load generator from {G_NET_PATH}')
    g_net.load_state_dict(torch.load(G_NET_PATH))
else:
    g_net.apply(weights_init)
print(g_net)

## 定義評論網路
d_net = NEPPDiscriminator().cuda()
if path.exists(D_NET_PATH):
    print(f'Load discriminator from {D_NET_PATH}')
    d_net.load_state_dict(torch.load(D_NET_PATH))
else:
    d_net.apply(weights_init)
print(d_net)

## 準備訓練資料
puipui_imgs = ImageFolder(root='puipui', transform=transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.RandomHorizontalFlip(0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

print('number of images:', len(puipui_imgs.imgs))

## 超參數
EPOCHS = 50000
BATCH_SIZE = 64
LR = 0.001
C = 0.01
UPDATE_G_PER_EPOCH = 5

## 訓練
dataloader = DataLoader(puipui_imgs, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)

g_optim = optimizers.RMSprop(g_net.parameters(), lr=LR)
d_optim = optimizers.RMSprop(d_net.parameters(), lr=LR)

fixed_noise = torch.randn(64, 100, 1, 1).cuda()

for epoch in range(1, EPOCHS+1):
    total_d_loss = 0
    total_g_loss = 0

    for i, batch in enumerate(dataloader):
        x = batch[0].cuda()
        x_size = x.size(0)
        random_noise = torch.randn(x_size, 100, 1, 1).cuda()

        # 用真和假的圖更新評論網路
        d_optim.zero_grad()

        # 真圖
        d_pred_real = d_net(x).view(-1)
        # 假圖
        fake_puipui = g_net(random_noise)
        d_pred_fake = d_net(fake_puipui.detach()).view(-1)

        d_loss = -torch.mean(d_pred_real) + torch.mean(d_pred_fake)
        d_loss.backward()
        d_optim.step()
        total_d_loss += d_loss.item()

        for p in d_net.parameters():
            p.data.clamp_(-C, C)

        # 更新生成網路
        if epoch % UPDATE_G_PER_EPOCH == 0:
            g_optim.zero_grad()
            d_pred = d_net(fake_puipui).view(-1)
            g_loss = -torch.mean(d_pred)
            g_loss.backward()
            g_optim.step()
            total_g_loss += g_loss.item()

            print(f'EPOCH: {epoch} | D loss: {total_d_loss} | G loss: {total_g_loss}')

    if epoch % 100 == 0:
        with torch.no_grad():
            generated_puipui = g_net(fixed_noise).detach().cpu()
            img = make_grid(generated_puipui, padding=2, normalize=True)
        plt.imsave(f'wgan_results/{epoch}.jpg', np.transpose(img,(1,2,0)).numpy())
        torch.save(g_net.state_dict(), G_NET_PATH)
        torch.save(d_net.state_dict(), D_NET_PATH)


