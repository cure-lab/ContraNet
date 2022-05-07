import torch.nn as nn
import torch.nn.functional as F
# normalize: 0.13, 0.31
# original input size: 28 x 28
# accept input size: 32 x 32 bilinear


class Mnist2LayerNet(nn.Module):
    def __init__(self):
        super(Mnist2LayerNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.mean(dim=-3, keepdim=True)
        x = F.interpolate(x, size=(28, 28), mode="bilinear",
                          align_corners=True)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    import torch
    from misc.load_dataset import LoadDataset
    from torch.utils.data import DataLoader
    from torchvision.transforms import Normalize
    test_data = LoadDataset(
        "MNIST", "./dataset", train=False, download=False,
        resize_size=32, hdf5_path=None, random_flip=False, norm=False)
    test_loader = DataLoader(
        test_data, batch_size=64, shuffle=False, num_workers=4,
        pin_memory=True)
    model = Mnist2LayerNet()
    weight = torch.load("pretrain/classifier/MNIST_Net.pth")
    model.load_state_dict(weight["model"])
    model = model.cuda()
    norm = Normalize(mean=[0.13, 0.13, 0.13], std=[0.31, 0.31, 0.31])
    cor, total = 0, 0
    for img, classId in test_loader:
        img = img.cuda()
        img = norm(img)
        classId = classId.cuda()
        logits = model(img)
        cor += (logits.argmax(dim=1) == classId).sum().item()
        total += img.shape[0]
    print("Accuracy: {:.2f}%".format(cor / total * 100))
