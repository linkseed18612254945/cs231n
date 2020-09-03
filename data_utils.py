from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

def cifar10_datasets(batch_size=32, train_sample_num=None, test_sample_num=None, flat_pix=True):
    nlp_data_path = '/home/ubuntu/likun/image_data'
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    train_data = datasets.CIFAR10(root=nlp_data_path, train=True)
    test_data = datasets.CIFAR10(root=nlp_data_path, train=False)

    if train_sample_num is not None:
        train_data.data = train_data.data[range(train_sample_num)]
        train_data.targets = train_data.targets[:train_sample_num]

    if test_sample_num is not None:
        test_data.data = test_data.data[range(test_sample_num)]
        test_data.targets = test_data.targets[:test_sample_num]

    if flat_pix:
        train_data.data = train_data.data.reshape(train_data.data.shape[0], -1)
        test_data.data = test_data.data.reshape(test_data.data.shape[0], -1)

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size)
    return train_data, test_data, train_dataloader, test_dataloader, classes

def show_data_example(dataset, classes, samples_per_class=7):
    num_classes = len(classes)
    label = np.array(dataset.targets)
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(label == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(dataset.data[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.savefig(f'{dataset.filename.split(".")[0]}_example.svg')

if __name__ == '__main__':
    train, test, _, _, classes = cifar10_datasets(32, 2000, 1000)
