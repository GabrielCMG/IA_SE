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
from torchsummary import summary


def imshow(img):
    """
    Fonction affichant une image.

    :param torch.Tensor img: image sous la forme d'un tenseur
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def datasets(batch_size=32, num_samples_subset=15000):
    """
    Fonction créant les datasets utilisés pour la classification. Le dataset d'entrainement est composé de 50000 images
    de CIFAR-10 et le dataset de test en est composé de 10000. La fonction propose également un sous-ensemble du dataset
    d'entrainement (il peut être utilisé pour faire un entrainement de durée plus courte).

    :param int batch_size: taille des batchs dans le dataset
    :param int num_samples_subset: nombre d'images dans le sous-ensemble du dataset d'entrainement
    :return: trainloader, sous-ensemble du trainloader, testloader et liste des classes
    """
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

    # Création des datasets
    trainloader = DataLoader(c10train, batch_size=batch_size, shuffle=True)
    trainloader_subset = DataLoader(c10train_subset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(c10test, batch_size=batch_size)

    # Liste des classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, trainloader_subset, testloader, classes


def entrainement(net, device, trainloader, testloader, n_epochs=70, lr=0.001):
    """
    Fonction effectuant l'entrainement d'un réseau de neurones. Elle affiche des informations à chaque EPOCH sur la loss
    et la précision du réseau sur le dataset d'entrainement et sur le dataset de test. La fonction affiche également
    régulièrement au cours des EPOCH des informations sur l'évolution de l'entrainement (loss et précision moyenne sur
    les images entrainées).

    :param nn.Module net: réseau ne neurones à entrainer
    :param torch.device device: appareil où les tenseurs seront alloués (GPU ou CPU)
    :param DataLoader trainloader: dataset d'entrainement
    :param DataLoader testloader: dataset de test
    :param int n_epochs: nombre d'EPOCH sur lequel l'entrainement sera réalisé
    :param float lr: learning rate de l'optimiseur
    :return: statistiques de l'entrainement
    """
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
    """
    Fonction sauvegardant le réseau entrainé ainsi que les statistiques de son entrainement.

    :param nn.Module net: réseau de neurones entrainé
    :param pd.DataFrame train_stats: statistiques de l'entrainement
    :param str nom_fichier: nom qualifiant le réseau de neurone qui sera présent dans les fichiers enregistrés
    """
    PATH = 'networks/densenet_' + nom_fichier + '.pth'
    torch.save(net.state_dict(), PATH)

    PATH2 = 'logs/data_' + nom_fichier + '.csv'
    train_stats.to_csv(PATH2, encoding='utf-8', index=False)


def pruning(net, amount_Conv2D, amount_Linear):
    """
    Fonction effectuant du pruning sur un réseau de neurone donné. Le pruning sera effectué sur les couches linéaires et
    les couches de convolution selon des proportions qui peuvent être différentes.

    :param nn.Module net: réseau de neurone sur lequel le pruning sera effectué
    :param float amount_Conv2D: proportion de neurone à désactiver sur les couches de convolution
    :param float amount_Linear: proportion de neurone à désactiver sur les couches linéaires
    """
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount_Conv2D)
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount_Linear)


def removePruning(net):
    """
    Fonction retirant le pruning qui a été effectué par la fonction pruning. Les paramètres ayant été mis à zéro par la
    fonction pruning restent à zéro.

    :param nn.Module net: réseau de neurone sur lequel le pruning sera retiré
    """
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, name='weight')
        elif isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight')


def printResults(train_stats):
    """
    Affiche le graphique précision et loss en fonction de l'EPOCH d'un entrainement à partir du DataFrame des
    statistiques de l'entrainement.

    :param pd.DataFrame train_stats: statistiques de l'entrainement
    """
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

    plt.title('Évolution de la loss et de la précision')

    fig1.legend()
    plt.show()


def printComparaison():
    """
    Fonction affichant un graphique précision en fonction de la taille du fichier. Les points sur le graphique
    correspondent aux différents entrainements réalisés pour ce projet. La taille des fichiers correspond à la taille
    des réseaux de neurone une fois compressés au format ZIP.
    """
    res = [[3920, 85.05, "DenseNet sur subset CIFAR-10"],
           [3923, 91.77, "DenseNet sur CIFAR-10 (config 1)"],
           [67127, 92.12, "DenseNet sur CIFAR-10 (config 2)"],
           [1208, 89.65, "DenseNet avec Convolution Groupée sur CIFAR-10"],
           [3036, 91.52, "DenseNet avec Depthwise Separation Convolutions sur CIFAR-10"],
           [1503, 91.36, "DenseNet avec Pruning sur CIFAR-10 (80%)"],
           [1095, 89.46, "DenseNet avec Pruning sur CIFAR-10 (90%)"],
           [537, 88.25, "DenseNet avec Convolution Groupée et Pruning sur CIFAR-10"],
           [1217, 89.77, "DenseNet avec Depthwise Separation Convolutions et Pruning sur CIFAR-10"]]

    f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')

    ax.axhline(y=90, color='gray', linestyle='--')
    ax2.axhline(y=90, color='gray', linestyle='--')
    for line in res:
        [x, y, label] = line
        if x < 5000:
            ax.scatter(x, y, marker='x', color='red')
            ax.text(x + .03, y + .03, label, fontsize=9)
        else:
            ax2.scatter(x, y, marker='x', color='red')
            ax2.text(x + .03, y + .03, label, fontsize=9)
    ax.set_xlim(0, 5000)
    ax2.set_xlim(65000, 70000)

    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.tick_right()

    d = .015
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    f.suptitle('Évolution de la précision en fonction de\nla taille du fichier compressé correspondant au réseau')
    f.supxlabel('Taille du fichier compressé correspondant au réseau [kB]')
    f.supylabel('Précision [%]')

    plt.show()


def predict(net, testloader, device):
    """
    Fonction effectuant une prédiction sur un dataset de test.

    :param nn.Module net: réseau de neurone entrainé
    :param testloader: dataset de test sur lequel la prédiction va être effectuée
    :param torch.device device: appareil où les tenseurs seront alloués (GPU ou CPU)
    """
    criterion = nn.CrossEntropyLoss()
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

    print(f"Test loss: {test_loss / len(testloader):.4f}.. "
          f"Test accuracy: {test_accuracy / len(testloader):.4f}.. ")


if __name__ == "__main__":
    # Définition de l'appareil où les tenseurs seront alloués
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # Création des différents datasets
    trainloader, trainloader_subset, testloader, classes = datasets()

    # Création du dataset
    net = DenseNetCifar().to(device)

    # Chargement d'un réseau de neurone pré-entrainé
    net.load_state_dict(torch.load('networks/densenet_naif.pth'))

    # Pruning sur le réseau de neurone
    pruning(net, 0.8, 0.8)

    # Entrainement du réseau de neurone
    train_stats = entrainement(net, device, trainloader, testloader, n_epochs=70)

    # Retrait du pruning sur le réseau de neurone
    removePruning(net)

    # Sauvegarde du réseau de neurone
    save(net, train_stats, "naif_pruning_80")

    # Affichage de l'évolution de la loss et de la précision de l'entrainement
    printResults(train_stats)

    # Affichage de la comparaison entre les différents réseaux entrainés
    printComparaison()

