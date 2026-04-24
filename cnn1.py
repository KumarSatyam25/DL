# ======================================
# 1. IMPORTS
# ======================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ======================================
# 2. LOAD DATA
# ======================================
transform = transforms.ToTensor()

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

# ======================================
# 3. MODEL (FIXED VERSION)
# ======================================
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(64, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)   # FIXED
        return self.classification_head(x)

model = CNNClassifier()

# ======================================
# 4. PARAMETER COUNT
# ======================================
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total Learnable Parameters:", total_params)

# ======================================
# 5. LOSS & OPTIMIZER
# ======================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ======================================
# 6. TRAINING
# ======================================
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ======================================
# 7. TESTING
# ======================================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# ======================================
# 8. CONFUSION MATRIX
# ======================================
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# ======================================
# 9. ACCURACY
# ======================================
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print("Test Accuracy:", accuracy)