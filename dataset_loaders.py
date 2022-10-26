import numpy as np
from torch.utils.data import random_split, Subset
from torchvision import transforms, datasets


def train_val_split(trainset, valset):
    val_length = int(0.1 * len(trainset))
    train_length = len(trainset) - val_length
    idx = list(range(len(trainset)))
    np.random.shuffle(idx)
    train_idx = idx[:train_length]
    val_idx = idx[train_length:]
    train = Subset(trainset, train_idx)
    val = Subset(valset, val_idx)
    return train, val


def load_mnist(data_root):
    ts = [
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ]

    transform = transforms.Compose(ts)
    trainset = datasets.MNIST(root=data_root,
                              train=True,
                              download=True,
                              transform=transform)

    testset = datasets.MNIST(root=data_root,
                             train=False,
                             download=True,
                             transform=transform)
    val_length = int(0.1 * len(trainset))

    train, val = random_split(
        trainset,
        lengths=[len(trainset) - val_length, val_length])
    return train, val, testset


def load_fashion_mnist(data_root):
    ts = [
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ]

    transform = transforms.Compose(ts)
    trainset = datasets.FashionMNIST(root=data_root,
                                     train=True,
                                     download=True,
                                     transform=transform)

    testset = datasets.FashionMNIST(root=data_root,
                                    train=False,
                                    download=True,
                                    transform=transform)
    val_length = int(0.1 * len(trainset))

    train, val = random_split(
        trainset,
        lengths=[len(trainset) - val_length, val_length])
    return train, val, testset


def load_CIFAR10(data_root):
    # validation = testing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root=data_root, train=True, download=True,
                                transform=transform_train)
    valset = datasets.CIFAR10(root=data_root, train=True, download=True,
                              transform=transform_test)

    testset = datasets.CIFAR10(root=data_root, train=False,
                               download=True, transform=transform_test)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # train, val = train_val_split(trainset, valset)
    return trainset, testset, testset


def load_CIFAR3(data_root):
    # validation = testing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root=data_root, train=True, download=True,
                                transform=transform_train)
    valset = datasets.CIFAR10(root=data_root, train=True, download=True,
                              transform=transform_test)

    testset = datasets.CIFAR10(root=data_root, train=False,
                               download=True, transform=transform_test,
                               )

    reduced_trainset = reduce_to_n_classes(trainset, 3)
    reduced_testset = reduce_to_n_classes(testset, 3)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # train, val = train_val_split(trainset, valset)
    return reduced_trainset, reduced_testset, reduced_testset


def reduce_to_n_classes(dataset, n):
    targets = np.array(dataset.targets)
    dataset.data = dataset.data[targets < n]
    dataset.targets = targets[targets < n]
    return dataset
