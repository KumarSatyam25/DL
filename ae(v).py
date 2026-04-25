# ==============================
# 1. IMPORTS
# ==============================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ==============================
# 2. DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# 3. DATA
# ==============================
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# ==============================
# 4. VAE MODEL
# ==============================
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(28*28, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28*28)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h))

    def forward(self, x):
        x = x.view(-1, 28*28)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# ==============================
# 5. LOSS FUNCTION
# ==============================
def vae_loss(recon_x, x, mu, logvar):
    x = x.view(-1, 28*28)

    # Reconstruction loss (BCE)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# ==============================
# 6. INITIALIZE
# ==============================
model = VAE(latent_dim=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ==============================
# 7. TRAINING
# ==============================
epochs = 10

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for images, _ in train_loader:
        images = images.to(device)

        recon, mu, logvar = model(images)
        loss = vae_loss(recon, images, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_loss/len(train_loader.dataset):.4f}")

# ==============================
# 8. GENERATE NEW DIGITS
# ==============================
model.eval()

with torch.no_grad():
    # Sample from standard normal distribution
    z = torch.randn(16, 20).to(device)
    samples = model.decode(z).cpu()

# ==============================
# 9. VISUALIZE GENERATED DIGITS
# ==============================
plt.figure(figsize=(6,6))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(samples[i].view(28,28), cmap='gray')
    plt.axis('off')

plt.suptitle("Generated Digits from VAE")
plt.show()