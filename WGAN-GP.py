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
n_critic = 5  # 判別器訓練次數
lambda_gp = 10  # 梯度懲罰係數
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 本機資料集與模型位置設定 ---
zip_path = r"C:\Users\ytyan\Desktop\Model code\HW2 GAN\img_align_celeba.zip"
extract_path = r"C:\Users\ytyan\Desktop\Model code\HW2 GAN\image\celeba\fake"
checkpoint_dir = r"C:\Users\ytyan\Desktop\Model code\HW2 GAN\checkpoint(wgan)"
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

# --- 建立 Generator  ---
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

# --- 建立 Critic ---
class Critic(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        
        )

    def forward(self, input):
        return self.main(input).view(-1)

# --- 梯度懲罰計算函數 ---
def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = critic(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# --- 初始化 ---
netG = Generator(z_dim).to(device)
netC = Critic().to(device)
optimizerC = optim.Adam(netC.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.999))
fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

# --- 訓練循環 ---
G_losses = []
C_losses = []

train_start_time = time.time()
print("Start WGAN-GP Training...")
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        
        # 訓練Critic
        netC.zero_grad()
        
        # 生成假圖
        z = torch.randn(real_images.size(0), z_dim, 1, 1, device=device)
        fake_images = netG(z)
        
        # 計算真實與假圖評分
        real_validity = netC(real_images)
        fake_validity = netC(fake_images.detach())
        
        # 計算梯度懲罰
        gradient_penalty = compute_gradient_penalty(netC, real_images.data, fake_images.data)
        
        # Critic損失計算
        c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        c_loss.backward()
        optimizerC.step()
        
        C_losses.append(c_loss.item())
        
        # 每n_critic次訓練生成器
        if i % n_critic == 0:
            netG.zero_grad()
            
            # 重新生成假圖
            fake_images = netG(z)
            g_loss = -torch.mean(netC(fake_images))
            
            g_loss.backward()
            optimizerG.step()
            
            G_losses.append(g_loss.item())
        
        # 輸出訓練狀態
        if i % 100 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"C_loss: {c_loss.item():.4f} G_loss: {g_loss.item():.4f}")
train_end_time = time.time()
total_train_time = train_end_time - train_start_time
print(f" Total Training Time: {total_train_time:.2f} seconds")

# --- 儲存模型與生成結果 ---
torch.save(netG.state_dict(), os.path.join(checkpoint_dir, 'wgan_gp_generator.pth'))
torch.save(netC.state_dict(), os.path.join(checkpoint_dir, 'wgan_gp_critic.pth'))

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
plt.savefig(os.path.join(checkpoint_dir, "real_vs_fake_wgan_gp.png"))
plt.show()

# --- 繪製損失曲線 ---
plt.figure(figsize=(10,5))
plt.plot(C_losses, label='Critic Loss')
plt.plot(G_losses, label='Generator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('WGAN-GP Training Loss')
plt.savefig(os.path.join(checkpoint_dir, 'wgan_gp_loss.png'))
plt.show()
