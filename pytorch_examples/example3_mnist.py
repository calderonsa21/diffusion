"""
Example 3 – MNIST Digits with DDPM (PyTorch, modern stack)

What it shows:
- DDPM can produce diverse, high-quality samples on real image data.
- Highlights computational cost and need for structured denoisers (CNNs).
- Demonstrates sampling inefficiency (T steps required for each sample).

Configuration notes:
- Uses 200 diffusion steps (reduced from 1000) for easier learning
- Trains for 30 epochs to ensure recognizable digit generation
- Model capacity increased (128 channels) for better expressiveness

Run:
  python pytorch_examples/example3_mnist.py
"""

import math
import os
from typing import Literal

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Positional / timestep embed
# -----------------------------
def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
  """Standard sinusoidal embedding."""
  half = dim // 2
  freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / (half - 1))
  args = timesteps.float()[:, None] * freqs[None, :]
  emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
  if dim % 2 == 1:
    emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
  return emb


# -----------------------------
# Schedules
# -----------------------------
def make_beta_schedule(num_steps: int, kind: Literal["linear", "cosine"] = "linear",
                       beta_start=1e-4, beta_end=0.02) -> torch.Tensor:
  if kind == "linear":
    return torch.linspace(beta_start, beta_end, num_steps)
  if kind == "cosine":
    # Cosine schedule from Nichol & Dhariwal (2021)
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps)
    alphas_cumprod = torch.cos(((x / num_steps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=0.999)
  raise ValueError(f"Unknown schedule: {kind}")


# -----------------------------
# Model: Lightweight U-Net style denoiser
# -----------------------------
class UNetDenoiser(nn.Module):
  """Lightweight convolutional denoiser for 28x28 images."""

  def __init__(self, channels=128, t_dim=64):
    super().__init__()
    self.t_dim = t_dim
    
    # Time embedding projection
    self.t_proj = nn.Linear(t_dim, channels)
    
    # Encoder
    self.conv1 = nn.Conv2d(1, channels // 2, 3, padding=1)
    self.conv2 = nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1)  # 14x14
    self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
    
    # Middle
    self.mid_conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    # Decoder
    self.conv4 = nn.Conv2d(channels, channels, 3, padding=1)
    self.conv5 = nn.Conv2d(channels, channels // 2, 3, padding=1)
    self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 14x14 -> 28x28
    self.conv6 = nn.Conv2d(channels // 2, channels // 2, 3, padding=1)
    self.out = nn.Conv2d(channels // 2, 1, 3, padding=1)

  def forward(self, x, t):
    # x: [B, 1, 28, 28]
    # t: [B]
    t_emb = sinusoidal_embedding(t, self.t_dim)
    t_proj = self.t_proj(t_emb)  # [B, channels]
    t_proj = t_proj[:, :, None, None]  # [B, channels, 1, 1]
    
    # Encoder
    h = F.relu(self.conv1(x))  # [B, channels//2, 28, 28]
    h = F.relu(self.conv2(h))  # [B, channels, 14, 14]
    h = F.relu(self.conv3(h))  # [B, channels, 14, 14]
    
    # Add time embedding
    h = h + t_proj
    
    # Middle
    h = F.relu(self.mid_conv(h))  # [B, channels, 14, 14]
    
    # Decoder
    h = F.relu(self.conv4(h))  # [B, channels, 14, 14]
    h = F.relu(self.conv5(h))  # [B, channels//2, 14, 14]
    h = self.upsample(h)  # [B, channels//2, 28, 28]
    h = F.relu(self.conv6(h))  # [B, channels//2, 28, 28]
    out = self.out(h)  # [B, 1, 28, 28]
    
    return out


# -----------------------------
# DDPM core (adapted for images)
# -----------------------------
class DDPM:
  def __init__(self, num_steps: int, schedule: Literal["linear", "cosine"] = "linear"):
    betas = make_beta_schedule(num_steps, kind=schedule).to(DEVICE)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    self.num_steps = num_steps
    self.betas = betas
    self.alphas = alphas
    self.alphas_cumprod = alphas_cumprod
    self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

  def q_sample(self, x0, t, noise=None):
    if noise is None:
      noise = torch.randn_like(x0)
    # Handle both 2D/3D and 4D tensors
    if len(x0.shape) == 4:  # Images: [B, C, H, W]
      sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
      sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    else:  # 2D/3D: [B, ...]
      sqrt_ac = self.sqrt_alphas_cumprod[t][:, None]
      sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None]
    return sqrt_ac * x0 + sqrt_om * noise

  def p_losses(self, model, x0, t):
    noise = torch.randn_like(x0)
    xt = self.q_sample(x0, t, noise)
    pred = model(xt, t)
    return F.mse_loss(pred, noise)

  @torch.no_grad()
  def p_sample(self, model, xt, t):
    # Handle both 2D/3D and 4D tensors
    if len(xt.shape) == 4:  # Images
      beta_t = self.betas[t][:, None, None, None]
      sqrt_one_minus_at = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
      sqrt_recip_alphas_t = 1.0 / torch.sqrt(self.alphas[t])[:, None, None, None]
    else:
      beta_t = self.betas[t][:, None]
      sqrt_one_minus_at = self.sqrt_one_minus_alphas_cumprod[t][:, None]
      sqrt_recip_alphas_t = 1.0 / torch.sqrt(self.alphas[t])[:, None]

    pred_noise = model(xt, t)
    mean = sqrt_recip_alphas_t * (xt - beta_t / sqrt_one_minus_at * pred_noise)
    if (t == 0).all():
      return mean
    noise = torch.randn_like(xt)
    sigma = torch.sqrt(beta_t)
    return mean + sigma * noise

  @torch.no_grad()
  def sample_loop(self, model, batch_size, shape):
    """Sample from the model. shape should be (C, H, W) for images."""
    if len(shape) == 3:  # Image: [C, H, W]
      xt = torch.randn(batch_size, *shape, device=DEVICE)
    else:  # 1D/2D
      xt = torch.randn(batch_size, *shape, device=DEVICE)
    
    for t in reversed(range(self.num_steps)):
      t_batch = torch.full((batch_size,), t, device=DEVICE, dtype=torch.long)
      xt = self.p_sample(model, xt, t_batch)
    return xt


# -----------------------------
# Data loading
# -----------------------------
def load_mnist(batch_size=128, data_dir="./data"):
  """Load MNIST dataset, normalized to [-1, 1]."""
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
  ])
  
  train_dataset = datasets.MNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transform
  )
  
  train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
  )
  
  return train_loader


# -----------------------------
# Visualization
# -----------------------------
def save_image_grid(images, nrow=8, title="", path=""):
  """Save a grid of images."""
  # images: [B, 1, 28, 28] or [B, 28, 28]
  if len(images.shape) == 4:
    images = images.squeeze(1)  # Remove channel dim
  
  # Denormalize from [-1, 1] to [0, 1]
  images = (images + 1.0) / 2.0
  images = torch.clamp(images, 0.0, 1.0)
  
  n = images.shape[0]
  nrow = min(nrow, n)
  ncol = (n + nrow - 1) // nrow
  
  fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 1.2, ncol * 1.2))
  if ncol == 1:
    axes = axes[None, :]
  elif nrow == 1:
    axes = axes[:, None]
  
  for i in range(n):
    row, col = i // nrow, i % nrow
    ax = axes[row, col]
    ax.imshow(images[i].cpu().numpy(), cmap='gray')
    ax.axis('off')
  
  # Hide extra subplots
  for i in range(n, nrow * ncol):
    row, col = i // nrow, i % nrow
    axes[row, col].axis('off')
  
  plt.suptitle(title, fontsize=12)
  plt.tight_layout()
  plt.savefig(path, dpi=150, bbox_inches='tight')
  plt.close()


# -----------------------------
# Training
# -----------------------------
def train(
    num_epochs=30,
    batch_size=128,
    num_steps=200,  # Reduced from 1000 for easier learning
    lr=2e-4,
    sample_every=5,  # Sample every N epochs
    num_samples=64,  # Number of samples to generate
    schedule="linear",
    out_dir="pytorch_examples_outputs"):
  os.makedirs(out_dir, exist_ok=True)

  print("Loading MNIST dataset...")
  train_loader = load_mnist(batch_size=batch_size)
  
  ddpm = DDPM(num_steps, schedule=schedule)
  model = UNetDenoiser().to(DEVICE)
  opt = torch.optim.Adam(model.parameters(), lr=lr)
  
  losses = []
  step = 0
  
  print(f"Training for {num_epochs} epochs...")
  for epoch in range(1, num_epochs + 1):
    epoch_losses = []
    
    for batch_idx, (x0, _) in enumerate(train_loader):
      x0 = x0.to(DEVICE)  # [B, 1, 28, 28]
      t = torch.randint(0, num_steps, (x0.shape[0],), device=DEVICE)
      
      loss = ddpm.p_losses(model, x0, t)
      opt.zero_grad()
      loss.backward()
      opt.step()
      
      losses.append(loss.item())
      epoch_losses.append(loss.item())
      step += 1
      
      if batch_idx % 100 == 0:
        print(f"Epoch {epoch}/{num_epochs}, batch {batch_idx}, loss={loss.item():.4f}")
    
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch} complete, avg loss={avg_loss:.4f}")
    
    # Sample and save at checkpoints
    if epoch % sample_every == 0 or epoch == num_epochs:
      print(f"Sampling {num_samples} images at epoch {epoch}...")
      with torch.no_grad():
        samples = ddpm.sample_loop(model, num_samples, shape=(1, 28, 28))
      
      sample_path = os.path.join(out_dir, f"example3_mnist_samples_epoch{epoch}.png")
      save_image_grid(
        samples,
        nrow=8,
        title=f"DDPM-generated MNIST digits after {epoch} epochs of training",
        path=sample_path
      )
      print(f"Saved samples to {sample_path}")

  # Save loss curve
  plt.figure(figsize=(10, 5))
  plt.plot(losses)
  plt.title("Training Loss")
  plt.xlabel("iteration")
  plt.ylabel("loss")
  plt.grid(True, alpha=0.3)
  loss_path = os.path.join(out_dir, "example3_mnist_loss_curve.png")
  plt.tight_layout()
  plt.savefig(loss_path, dpi=200)
  print(f"Saved loss curve to {loss_path}")


if __name__ == "__main__":
  train()

