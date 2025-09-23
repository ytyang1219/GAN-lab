
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
epochs = 20
image_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 本機資料集與模型位置設定 ---
zip_path = r"C:\Users\ytyan\Desktop\Model code\HW2 GAN\img_align_celeba.zip"
extract_path = r"C:\Users\ytyan\Desktop\Model code\HW2 GAN\image\celeba\fake"
checkpoint_dir =r"C:\Users\ytyan\Desktop\Model code\HW2 GAN\checkpoint"
os.makedirs(extract_path, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# --- 若資料夾內尚未解壓，進行解壓 ---
if not os.path.exists(os.path.join(extract_path, 'img_align_celeba')):
    print("解壓縮資料集...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

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
            nn.LeakyReLU(0.2, inplace=True),  
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),  
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),  
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True), 
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, input):
        return self.main(input)

# --- 建立 Discriminator ---

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

# --- 儲存模型 ---
def saveModel(modelG, lossG):
    save_path = os.path.join(checkpoint_dir, 'dcgan_generator_best.pth')
    torch.save(modelG.state_dict(), save_path)
    print(f"Generator saved at loss {lossG:.4f} to {save_path}")

# --- 初始化 ---
netG = Generator(z_dim).to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0003, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.999))
fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)
best_G_loss = float('inf')
G_losses, D_losses = [], []

train_start_time = time.time()

# --- 訓練開始 ---
print("Start Training...")
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        real_images = real_images + torch.randn_like(real_images) * 0.03  # 3%噪聲強度

        b_size = real_images.size(0)
        labels_real = torch.ones(b_size, device=device)
        labels_fake = torch.zeros(b_size, device=device)

        # 訓練 Discriminator
        netD.zero_grad()
        output_real = netD(real_images)
        lossD_real = criterion(output_real, labels_real)

        noise = torch.randn(b_size, z_dim, 1, 1, device=device)
        fake_images = netG(noise)
        output_fake = netD(fake_images.detach())
        lossD_fake = criterion(output_fake, labels_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # 訓練 Generator
        netG.zero_grad()
        output = netD(fake_images)
        lossG = criterion(output, labels_real)
        lossG.backward()
        optimizerG.step()

        G_losses.append(lossG.item())
        D_losses.append(lossD.item())

        if i % 100 == 0:
            print(f"[{epoch+1}/{epochs}] [{i}/{len(dataloader)}] Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")

        if lossG.item() < best_G_loss:
            best_G_loss = lossG.item()
            saveModel(netG, best_G_loss)

train_end_time = time.time()
total_train_time = train_end_time - train_start_time
print(f" Total Training Time: {total_train_time:.2f} seconds")

# --- 畫圖與輸出 ---
netG.eval() 
with torch.no_grad():
    infer_start_time = time.time()
    fake = netG(fixed_noise).detach().cpu()
    infer_end_time = time.time()

    avg_infer_time_ms = (infer_end_time - infer_start_time) / fake.size(0) * 1000
    print(f"Average Inference Time per image: {avg_infer_time_ms:.4f} ms")

real_images = next(iter(dataloader))[0][:64]
grid_real = make_grid(real_images, padding=2, normalize=True)
grid_fake = make_grid(fake, padding=2, normalize=True)

# === 畫 Loss 曲線並儲存 ===
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

loss_curve_path = os.path.join(checkpoint_dir, "loss_curve.png")
plt.savefig(loss_curve_path)
print(f"Loss 曲線儲存至：{loss_curve_path}")
plt.show()
# === 畫 Real vs Fake 圖片並儲存 ===
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(grid_real, (1, 2, 0)))

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(grid_fake, (1, 2, 0)))

real_fake_path = os.path.join(checkpoint_dir, "real_vs_fake.png")
plt.savefig(real_fake_path)
print(f"Real vs Fake 圖像儲存至：{real_fake_path}")
plt.show()



# 最終儲存
torch.save(netG.state_dict(), os.path.join(checkpoint_dir, 'dcgan_generator_final.pth'))
torch.save(netD.state_dict(), os.path.join(checkpoint_dir, 'dcgan_discriminator_final.pth'))
