# imports
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import netwoks
from DenseNet import DenseNetCifar, DenseNet201
from DenseNetGroupedConv import DenseNetCifarGroupedConv
from DenseNetDepthwiseSeparable import DenseNetCifarDepthwiseSeparable

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import prune
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def datasets(batch_size=32, num_samples_subset=15000):
    normalize_scratch = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_scratch,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize_scratch,
    ])

    rootdir = 'data'

    c10train = CIFAR10(rootdir, train=True, download=True, transform=transform_train)
    c10test = CIFAR10(rootdir, train=False, download=True, transform=transform_test)

    num_train_examples = len(c10train)

    seed = 2147483647
    indices = list(range(num_train_examples))
    np.random.RandomState(seed=seed).shuffle(indices)

    c10train_subset = torch.utils.data.Subset(c10train, indices[:num_samples_subset])

    print(f"Initial CIFAR10 dataset has {len(c10train)} samples")
    print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")

    trainloader = DataLoader(c10train, batch_size=batch_size, shuffle=True)
    trainloader_subset = DataLoader(c10train_subset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(c10test, batch_size=batch_size)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, trainloader_subset, testloader, classes


def entrainement(net, device, trainloader, testloader, n_epochs=70, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_stats = pd.DataFrame(
        columns=['Epoch', 'Time per epoch', 'Avg time per step', 'Train loss', 'Train accuracy', 'Test loss',
                 'Test accuracy'])

    running_loss = 0
    for epoch in range(n_epochs):

        since = time.time()

        steps = 0
        train_accuracy = 0
        for i, data in enumerate(trainloader, 0):
            steps += 1

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # calculate train top-1 accuracy
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            if i % 100 == 0:
                print(
                    f'Epoch: {epoch + 1}/{n_epochs} (Step {i + 1}/{len(trainloader)}) | loss: {running_loss / steps:.4f}, accuracy: {train_accuracy / steps:.4f}')

        time_elapsed = time.time() - since

        test_loss = 0
        test_accuracy = 0
        net.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                batch_loss = criterion(outputs, labels)

                test_loss += batch_loss.item()

                # Calculate test top-1 accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch + 1}/{n_epochs}.. "
              f"Time per epoch: {time_elapsed:.4f}.. "
              f"Average time per step: {time_elapsed / len(trainloader):.4f}.. "
              f"Train loss: {running_loss / len(trainloader):.4f}.. "
              f"Train accuracy: {train_accuracy / len(trainloader):.4f}.. "
              f"Test loss: {test_loss / len(testloader):.4f}.. "
              f"Test accuracy: {test_accuracy / len(testloader):.4f}.. ")

        train_stats_new_row = pd.DataFrame(
            {'Epoch': epoch, 'Time per epoch': time_elapsed,
             'Avg time per step': time_elapsed / len(trainloader),
             'Train loss': running_loss / len(trainloader),
             'Train accuracy': train_accuracy / len(trainloader), 'Test loss': test_loss / len(testloader),
             'Test accuracy': test_accuracy / len(testloader)}, index=[0])

        train_stats = pd.concat([train_stats, train_stats_new_row])

        running_loss = 0
        net.train()

    return train_stats


def save(net, train_stats, nom_fichier):
    PATH = 'networks/densenet_' + nom_fichier + '.pth'
    torch.save(net.state_dict(), PATH)

    PATH2 = 'logs/data_' + nom_fichier + '.csv'
    train_stats.to_csv(PATH2, encoding='utf-8', index=False)


def pruning(net, amount_Conv2D, amount_Linear):
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount_Conv2D)
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount_Linear)


def unpruning(net):
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, name='weight')
        elif isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight')


def print_results(train_stats):
    train_stats = train_stats.to_numpy()

    epochs = train_stats[:, 0]
    acc_train = train_stats[:, 4]
    acc_test = train_stats[:, 6]
    loss_train = train_stats[:, 3]
    loss_test = train_stats[:, 5]

    fig1, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(epochs, acc_train, 'g--', label='Train accuracy')
    ax1.plot(epochs, acc_test, 'b--', label='Test accuracy')
    ax2.plot(epochs, loss_train, 'g-', label='Train loss')
    ax2.plot(epochs, loss_test, 'b-', label='Test loss')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Loss')

    fig1.legend()
    plt.show()


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    trainloader, trainloader_subset, testloader, classes = datasets()

    net = DenseNet201().to(device)

    train_stats = entrainement(net, device, trainloader, testloader, n_epochs=50)

    save(net, train_stats, "test_201")

    print_results(train_stats)
