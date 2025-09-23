# --- 匯入必要套件 ---
import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# --- 基本設定 ---
batch_size = 128
z_dim = 100
epochs = 50
image_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 本機資料集與模型位置設定 ---
zip_path = r"C:\Users\ytyan\Desktop\Model code\HW2 GAN\img_align_celeba.zip"
extract_path = r"C:\Users\ytyan\Desktop\Model code\HW2 GAN\image\celeba\fake"
checkpoint_dir = r"C:\Users\ytyan\Desktop\Model code\HW2 GAN\checkpoint(ls)"
os.makedirs(extract_path, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# --- 資料轉換與載入 ---
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
dataset = datasets.ImageFolder(
    root=r"C:\Users\ytyan\Desktop\Model code\HW2 GAN\image\celeba",
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 建立 Generator ---
class Generator(nn.Module):
    def __init__(self, z_dim, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# --- 建立 Discriminator (移除最後的Sigmoid) ---
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
            # 移除Sigmoid，使用線性輸出
        )

    def forward(self, input):
        return self.main(input).view(-1)

# --- 初始化 ---
netG = Generator(z_dim).to(device)
netD = Discriminator().to(device)
criterion = nn.MSELoss() 

# 調整學習率：判別器學習率降低，生成器學習率提高
optimizerD = optim.Adam(netD.parameters(), lr=0.00005, betas=(0.5, 0.999), weight_decay=0.0001)
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

G_losses = []
D_losses = []

train_start_time = time.time()
print("Start LSGAN Training...")
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        
        # 添加噪聲到真實圖像
        real_images = real_images + 0.05 * torch.randn_like(real_images)
        
        b_size = real_images.size(0)
        
        # 標籤平滑化
        real_labels = torch.ones(b_size, device=device) * 0.9  # 從1.0改為0.9
        fake_labels = torch.zeros(b_size, device=device)

        # 訓練 Discriminator (每2次迭代訓練一次)
        if i % 2 == 0:
            netD.zero_grad()
            output_real = netD(real_images)
            lossD_real = criterion(output_real, real_labels)
            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach())
            lossD_fake = criterion(output_fake, fake_labels)
            lossD = 0.5 * (lossD_real + lossD_fake)
            lossD.backward()
            optimizerD.step()
            D_losses.append(lossD.item())
        
        # 訓練 Generator (每次迭代都訓練)
        netG.zero_grad()
        noise = torch.randn(b_size, z_dim, 1, 1, device=device)
        fake_images = netG(noise)
        output = netD(fake_images)
        # 生成器目標：判別器輸出接近0.9，不是1.0
        lossG = criterion(output, real_labels)
        lossG.backward()
        optimizerG.step()
        G_losses.append(lossG.item())

        if i % 100 == 0:
            print(f"[{epoch+1}/{epochs}] [{i}/{len(dataloader)}] Loss_D: {lossD.item() if i % 2 == 0 else 'N/A':.4f} Loss_G: {lossG.item():.4f}")

train_end_time = time.time()
total_train_time = train_end_time - train_start_time
print(f" Total Training Time: {total_train_time:.2f} seconds")

# --- 儲存模型與生成結果 ---
torch.save(netG.state_dict(), os.path.join(checkpoint_dir, 'lsgan_generator.pth'))
torch.save(netD.state_dict(), os.path.join(checkpoint_dir, 'lsgan_discriminator.pth'))

with torch.no_grad():
    infer_start_time = time.time()
    fake = netG(fixed_noise).detach().cpu()
    infer_end_time = time.time()
    avg_infer_time_ms = (infer_end_time - infer_start_time) / fake.size(0) * 1000
    print(f"Average Inference Time per image: {avg_infer_time_ms:.4f} ms")

# 取得一批真實圖片
real_images = next(iter(dataloader))[0][:64].cpu()
grid_real = make_grid(real_images, padding=2, normalize=True)

# 生成一批假圖片
netG.eval()
with torch.no_grad():
    fake_images = netG(fixed_noise).detach().cpu()
    grid_fake = make_grid(fake_images, padding=2, normalize=True)

# 畫圖
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(grid_real, (1, 2, 0)))

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(grid_fake, (1, 2, 0)))

plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, "real_vs_fake_lsgan.png"))
plt.show()

# --- 繪製損失曲線 ---
plt.figure(figsize=(10,5))
plt.plot(G_losses, label='Generator Loss')
plt.plot(D_losses, label='Discriminator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('LSGAN Training Loss')
plt.savefig(os.path.join(checkpoint_dir, 'lsgan_loss.png'))
plt.show()
