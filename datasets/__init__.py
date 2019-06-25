
# 用 pytorch 加载数据集

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


class MNIST(Dataset):
    """
    类似 torchvision.datasets.ImageFolder 的Dataset
    """
    def __init__(self, root, transform=None, preload=False):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform
        
        root_path = Path(root)
        for fn in root_path.rglob('*.png'):
            label = fn.parts[-2]
            self.filenames.append((fn, int(label)))
            
        # if preload dataset into memory
        if preload:
            self._preload()
            
        self.len = len(self.filenames)
                              
    def _preload(self):
        """
        Preload dataset to memory
        """
        self.labels = []
        self.images = []
        for image_fn, label in self.filenames:            
            # load images
            image = Image.open(image_fn)
            self.images.append(image.copy())
            # avoid too many opened files bug
            image.close()
            self.labels.append(label)

    # probably the most important to customize.
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
            label = self.labels[index]
        else:
            # If on-demand data loading
            image_fn, label = self.filenames[index]
            image = Image.open(image_fn)
            
        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            image = self.transform(image)
        # return image and label
        return image, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
