"""
Minimal DDPM example in PyTorch: 1D Gaussian mixture.
Run: python pytorch_examples/example1_1d_gaussians.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
  half = dim // 2
  freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / (half - 1))
  args = timesteps.float()[:, None] * freqs[None, :]
  emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
  if dim % 2 == 1:
    emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
  return emb


def make_beta_schedule(num_steps: int, beta_start=1e-4, beta_end=0.02):
  return torch.linspace(beta_start, beta_end, num_steps)


class MLPDenoiser(nn.Module):
  def __init__(self, hidden=128, t_dim=32):
    super().__init__()
    self.t_dim = t_dim
    self.fc1 = nn.Linear(1 + t_dim, hidden)
    self.fc2 = nn.Linear(hidden, hidden)
    self.out = nn.Linear(hidden, 1)

  def forward(self, x, t):
    t_emb = sinusoidal_embedding(t, self.t_dim)
    h = torch.cat([x, t_emb], dim=-1)
    h = F.relu(self.fc1(h))
    h = F.relu(self.fc2(h))
    return self.out(h)


class DDPM:
  def __init__(self, num_steps: int):
    betas = make_beta_schedule(num_steps).to(DEVICE)
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
    xt = torch.randn(batch_size, 1, device=DEVICE)
    for t in reversed(range(self.num_steps)):
      t_batch = torch.full((batch_size,), t, device=DEVICE, dtype=torch.long)
      xt = self.p_sample(model, xt, t_batch)
    return xt


def sample_data(batch_size: int):
  centers = torch.tensor([-2.0, 2.0]).to(DEVICE)
  choices = torch.randint(0, 2, (batch_size,), device=DEVICE)
  base = centers[choices]
  noise = torch.randn(batch_size, device=DEVICE) * 0.3
  x = base + noise
  return x[:, None]  # [B, 1]


def train(
    steps=2000,
    batch_size=256,
    num_steps=200,
    lr=1e-3,
    log_every=200,
    sample_size=2000,
    out_dir="pytorch_examples_outputs"):
  os.makedirs(out_dir, exist_ok=True)
  ddpm = DDPM(num_steps)
  model = MLPDenoiser().to(DEVICE)
  opt = torch.optim.Adam(model.parameters(), lr=lr)
  losses = []
  for step in range(1, steps + 1):
    x0 = sample_data(batch_size)
    t = torch.randint(0, num_steps, (batch_size,), device=DEVICE)
    loss = ddpm.p_losses(model, x0, t)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if step % log_every == 0:
      print(f"step {step:04d} loss={loss.item():.4f}")

  with torch.no_grad():
    samples = ddpm.sample_loop(model, sample_size).cpu().numpy().reshape(-1)

  # Plot
  plt.figure(figsize=(10, 4))
  plt.subplot(1, 2, 1)
  plt.plot(losses)
  plt.title("Training loss")
  plt.xlabel("step")
  plt.ylabel("loss")

  plt.subplot(1, 2, 2)
  plt.hist(samples, bins=50, density=True, alpha=0.8, label="model")
  plt.hist(sample_data(sample_size).cpu().numpy().reshape(-1),
           bins=50, density=True, alpha=0.4, label="data")
  plt.legend()
  plt.title("1D mixture")

  out_path = os.path.join(out_dir, "example1_hist.png")
  plt.tight_layout()
  plt.savefig(out_path, dpi=200)
  print(f"Saved plot to {out_path}")


if __name__ == "__main__":
  train()

