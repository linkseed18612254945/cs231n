from torchvision import datasets
from tqdm import trange

def cifar10_datasets():
    nlp_data_path = '/home/ubuntu/likun/image_data'
    train_data = datasets.CIFAR10(root=nlp_data_path, train=True)
    test_data = datasets.CIFAR10(root=nlp_data_path, train=False)
    return train_data, test_data
