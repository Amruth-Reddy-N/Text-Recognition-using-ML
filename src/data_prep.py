import numpy as np
from torchvision import datasets, transforms

def load_emnist(split='balanced', root='./data'):
    transform = transforms.Compose([transforms.ToTensor()])
    ds_train = datasets.EMNIST(root=root, split=split, train=True, download=True, transform=transform)
    ds_test = datasets.EMNIST(root=root, split=split, train=False, download=True, transform=transform)
    return ds_train, ds_test

def tensor_to_numpy_img(tensor):
    arr = tensor.numpy().squeeze(0)
    arr = np.rot90(arr, k=3)
    arr = np.fliplr(arr)
    return arr

def make_flat_features(dataset):
    X = []
    y = []
    for img, label in dataset:
        arr = tensor_to_numpy_img(img)
        X.append(arr.flatten())
        y.append(label)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)
