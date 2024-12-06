import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


# Define a simple 3-layer DNN with convolutional and fully connected layers

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Initial convolution block
        self.conv_initial = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(num_features=128)
        self.channel_reducer1 = nn.Conv2d(in_channels=128, out_channels=4, kernel_size=1, padding=1)

        # First feature extraction block
        self.conv_block1_1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.bn_block1_1 = nn.BatchNorm2d(num_features=16)
        self.pool_block1 = nn.MaxPool2d(2, 2)
        
        # Second feature extraction block
        self.conv_block2_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn_block2_1 = nn.BatchNorm2d(num_features=16)
        self.conv_block2_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn_block2_2 = nn.BatchNorm2d(num_features=32)

        # Dimensionality reduction block
        self.pool_block2 = nn.MaxPool2d(2, 2)
        self.channel_reducer2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=1)

        # Final feature extraction block
        self.conv_final1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn_final1 = nn.BatchNorm2d(num_features=16)
        self.conv_final2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn_final2 = nn.BatchNorm2d(num_features=32)
        
        # Classification head
        self.classifier = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1, padding=1)
        self.global_pool = nn.AvgPool2d(kernel_size=7)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        # Initial convolution block
        x = self.conv_initial(x)
        x = F.relu(self.bn_initial(x))
        x = self.dropout(x)
        x = self.channel_reducer1(x)

        # First feature extraction block
        x = self.conv_block1_1(x)
        x = F.relu(self.bn_block1_1(x))
        x = self.dropout(x)
        x = self.pool_block1(x)

        # Second feature extraction block
        x = self.conv_block2_1(x)
        x = F.relu(self.bn_block2_1(x))
        x = self.dropout(x)
        x = self.conv_block2_2(x)
        x = F.relu(self.bn_block2_2(x))
        x = self.dropout(x)

        # Dimensionality reduction block
        x = self.pool_block2(x)
        x = self.channel_reducer2(x)

        # Final feature extraction block
        x = self.conv_final1(x)
        x = F.relu(self.bn_final1(x))
        x = self.dropout(x)
        x = self.conv_final2(x)
        x = F.relu(self.bn_final2(x))
        x = self.dropout(x)

        # Classification head
        x = self.classifier(x)
        x = self.global_pool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


# Transformations and data loading
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Instantiate model, loss function, and optimizer
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


torch.manual_seed(1)
batch_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)


model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 2):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    

# Save the model as 'model_latest.pth'
torch.save(model.state_dict(), 'model_latest.pth')
