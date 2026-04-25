import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# DATA
# ==============================
transform = transforms.ToTensor()

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ==============================
# MODEL
# ==============================
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# ==============================
# LOAD MODEL (IMPORTANT)
# ==============================
model = Autoencoder(latent_dim=32).to(device)

# If you trained earlier, load weights:
# model.load_state_dict(torch.load("autoencoder.pth"))

# ==============================
# LATENT EXTRACTION (YOUR CODE)
# ==============================
model.eval()
latents = []
labels = []

with torch.no_grad():
    for images, lbls in test_loader:
        images = images.to(device)
        _, z = model(images)
        
        latents.append(z.cpu())
        labels.append(lbls)

latents = torch.cat(latents)
labels = torch.cat(labels)

print("Latent shape:", latents.shape)
print("Labels shape:", labels.shape)