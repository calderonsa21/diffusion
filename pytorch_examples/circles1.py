"""
Example 2 – 2D Noisy Circle with DDPM (PyTorch, modern stack)

What it shows:
- DDPM can learn manifold-like structure (points concentrated near a 1D curve in 2D).
- Illustrates the "Euclidean noise + manifold data" assumption.

Run:
  python pytorch_examples/circles1.py
"""

import math
import os
from typing import Literal

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# Model
# -----------------------------
class MLPDenoiser2D(nn.Module):
  """Predicts 2D noise epsilon(x_t, t) for 2D input."""

  def __init__(self, hidden=256, t_dim=64):
    super().__init__()
    self.t_dim = t_dim
    # Input: 2D x_t + time embedding
    self.fc1 = nn.Linear(2 + t_dim, hidden)
    self.fc2 = nn.Linear(hidden, hidden)
    self.fc3 = nn.Linear(hidden, hidden)
    self.out = nn.Linear(hidden, 2)  # Output 2D noise

  def forward(self, x, t):
    t_emb = sinusoidal_embedding(t, self.t_dim)
    h = torch.cat([x, t_emb], dim=-1)
    h = F.relu(self.fc1(h))
    h = F.relu(self.fc2(h))
    h = F.relu(self.fc3(h))
    return self.out(h)


# -----------------------------
# DDPM core
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
  def sample_loop(self, model, batch_size):
    xt = torch.randn(batch_size, 2, device=DEVICE)  # 2D noise
    for t in reversed(range(self.num_steps)):
      t_batch = torch.full((batch_size,), t, device=DEVICE, dtype=torch.long)
      xt = self.p_sample(model, xt, t_batch)
    return xt


# -----------------------------
# Data
# -----------------------------
def sample_circle_data(n: int, radius=1.0, noise_std=0.1) -> torch.Tensor:
  """
  Generate points on a circle with small radial noise.
  
  Args:
    n: Number of samples
    radius: Circle radius (default 1.0)
    noise_std: Standard deviation of radial Gaussian noise
  
  Returns:
    Tensor of shape [n, 2] with 2D points
  """
  # Sample angles uniformly from [0, 2π]
  angles = torch.rand(n, device=DEVICE) * 2 * math.pi
  
  # Points on circle: (r cos θ, r sin θ)
  x = radius * torch.cos(angles)
  y = radius * torch.sin(angles)
  
  # Add small Gaussian noise
  noise = torch.randn(n, 2, device=DEVICE) * noise_std
  points = torch.stack([x, y], dim=1) + noise
  
  return points


# -----------------------------
# Training / plotting
# -----------------------------
def train(
    steps=8000,
    batch_size=256,
    num_steps=1000,
    lr=1e-3,
    log_every=500,
    sample_size=5000,
    schedule="linear",
    out_dir="pytorch_examples_outputs"):
  os.makedirs(out_dir, exist_ok=True)

  # Prepare dataset (50k points as specified)
  data = sample_circle_data(50_000)
  dataset = torch.utils.data.TensorDataset(data)
  loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
  data_iter = iter(loader)

  ddpm = DDPM(num_steps, schedule=schedule)
  model = MLPDenoiser2D().to(DEVICE)
  opt = torch.optim.Adam(model.parameters(), lr=lr)
  losses = []

  print("Training denoiser on 2D circle data...")
  for step in range(1, steps + 1):
    try:
      (x0,) = next(data_iter)
    except StopIteration:
      data_iter = iter(loader)
      (x0,) = next(data_iter)
    x0 = x0.to(DEVICE)
    t = torch.randint(0, num_steps, (batch_size,), device=DEVICE)
    loss = ddpm.p_losses(model, x0, t)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if step % log_every == 0:
      print(f"step {step:05d} loss={loss.item():.4f}")

  print("Sampling from model...")
  with torch.no_grad():
    samples = ddpm.sample_loop(model, sample_size).cpu().numpy()

  # Generate fresh data for comparison
  data_samples = sample_circle_data(sample_size).cpu().numpy()

  # Plot 1: Training data scatter
  plt.figure(figsize=(6, 6))
  plt.scatter(data_samples[:, 0], data_samples[:, 1], s=4, alpha=0.5, label="training data")
  plt.xlabel("x₁")
  plt.ylabel("x₂")
  plt.title("2D Noisy Circle: Training Data")
  plt.axis("equal")
  plt.legend()
  plt.grid(True, alpha=0.3)
  data_path = os.path.join(out_dir, "example2_scatter_data.png")
  plt.tight_layout()
  plt.savefig(data_path, dpi=200)
  print(f"Saved training data plot to {data_path}")

  # Plot 2: Generated samples scatter
  plt.figure(figsize=(6, 6))
  plt.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.5, label="generated samples", color="orange")
  plt.xlabel("x₁")
  plt.ylabel("x₂")
  plt.title("2D Noisy Circle: Generated Samples")
  plt.axis("equal")
  plt.legend()
  plt.grid(True, alpha=0.3)
  model_path = os.path.join(out_dir, "example2_scatter_model.png")
  plt.tight_layout()
  plt.savefig(model_path, dpi=200)
  print(f"Saved generated samples plot to {model_path}")

  # Plot 3: Loss curve (optional)
  plt.figure(figsize=(8, 5))
  plt.plot(losses)
  plt.title("Training Loss")
  plt.xlabel("iteration")
  plt.ylabel("loss")
  plt.grid(True, alpha=0.3)
  loss_path = os.path.join(out_dir, "example2_loss_curve.png")
  plt.tight_layout()
  plt.savefig(loss_path, dpi=200)
  print(f"Saved loss curve to {loss_path}")


if __name__ == "__main__":
  train()

