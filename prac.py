import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 32 * 8 * 8)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ArtificialDataset(Dataset):
    def __init__(self,data):
        self.images=data["images"]
        self.labels=data["labels"]
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        image=self.images[idx]
        label=self.labels[idx]
        return image,label


data1= torch.load("artificial_data.pt")
dataset1=ArtificialDataset(data1)

train_size=int(0.8*len(dataset1))
test_size=len(dataset1)-train_size
train_dataset,test_dataset=random_split(dataset1,[train_size,test_size])
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=CNN().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.001)

patience=5
best_loss=float('inf')
counter=0

epochs=50
for epoch in range(50):
    model.train()
    train_loss=0
    correct=0
    total=0

    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    train_accuracy=100*correct/total
    print(f'Epoch {epoch+1}, Loss: {train_loss/len(train_loader)}, Accuracy: {train_accuracy:.2f}%')

    model.eval()
    test_loss=0
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels)
            test_loss+=loss.item()
            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    test_accuracy=100*correct/total
    print(f'Test Loss: {test_loss/len(test_loader)}, Test Accuracy: {test_accuracy:.2f}%')

    if test_loss<best_loss:
        best_loss=test_loss
        counter=0
    else:
        counter+=1
        if counter>=patience:
            print("Early stopping")
            break

data2= torch.load("artificial_data2.pt")
dataset2=ArtificialDataset(data2)
test_loader2=DataLoader(dataset2,batch_size=64,shuffle=False)

model=CNN().to(device)
model.load_state_dict(torch.load("cnn_model.pth"))

for param in model.parameters():
    param.requires_grad=False

model.fc2=nn.Linear(128,10).to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.fc2.parameters(),lr=0.001)

epochs=20
for epoch in range(epochs):
    model.train()
    train_loss=0
    correct=0
    total=0

    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    train_accuracy=100*correct/total
    print(f'Epoch {epoch+1}, Loss: {train_loss/len(train_loader)}, Accuracy: {train_accuracy:.2f}%')

    
