import mf
mf.config(framework='torch') # noqa

from mf import nn
from mf import Tensor
from mf import optimizer
from mf import data

import numpy as np
from tqdm import tqdm
from mnist import mnist


class MNISTDataset(data.Dataset):
    try:
        x_train, t_train, x_test, t_test = mnist.load()
    except Exception:  # noqa
        mnist.init()
        x_train, t_train, x_test, t_test = mnist.load()

    def __init__(self, train=True):
        super().__init__()
        self.train = train

    def __getitem__(self, index):
        if self.train:
            return Tensor(MNISTDataset.x_train[index].astype("float32") / 255.0), \
                Tensor(np.array((MNISTDataset.t_train[index],))).long()
        else:
            return Tensor(MNISTDataset.x_test[index].astype("float32") / 255.0), \
                Tensor(np.array((MNISTDataset.t_test[index],))).long()

    def __len__(self):
        if self.train:
            return MNISTDataset.x_train.shape[0]
        else:
            return MNISTDataset.x_test.shape[0]


train_dataset = MNISTDataset(train=True)
test_dataset = MNISTDataset(train=True)

train_dataloader = data.DataLoader(train_dataset, batch_size=128)
test_dataloader = data.DataLoader(test_dataset, batch_size=128)


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2D(64, kernel_size=3),
            nn.BatchNorm2D(),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2),

            nn.Conv2D(32, kernel_size=3),
            nn.BatchNorm2D(),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2),

            nn.Conv2D(16, kernel_size=3),
            nn.BatchNorm2D(),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2)
        )
        self.fc = nn.Linear(num_classes)

    def forward(self, x):
        x = x.reshape([-1, 1, 28, 28])  # BCWH
        x = self.seq(x)
        x = x.reshape([-1, 16 * 3 * 3])
        x = self.fc(x)
        return mf.softmax(x, dim=1)


model = ConvNet()
model = model.to("cuda")

optim = optimizer.Adam(model.parameters, learning_rate=3e-4)

model.train()
for epoch in range(3):
    for data in tqdm(train_dataloader):
        x, t = data
        x, t = x.to("cuda"), t.to("cuda")
        x = x.reshape([-1, 1, 28, 28])
        y = model(x)

        t = t.squeeze()

        loss = mf.nn.functional.cross_entropy(y, t)

        optim.zero_grad()
        loss.backward()
        optim.step()


model.eval()

total_count = 0
correct_count = 0
with mf.no_grad():
    for data in tqdm(test_dataloader):
        x, t = data
        x, t = x.to("cuda"), t.to("cuda")
        t = t.squeeze()
        x = x.reshape([-1, 1, 28, 28])

        y = model(x)

        correct_count += (y.argmax(dim=1) == t).sum().item()

        total_count += y.shape[0]

accuracy = correct_count / total_count
print(f"Accuracy: {accuracy}")
