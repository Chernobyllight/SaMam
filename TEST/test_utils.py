import random
import warnings
from pathlib import Path

from PIL import Image
from torch.utils.data import IterableDataset, Dataset
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop
from torchvision.utils import save_image

Image.MAX_IMAGE_PIXELS = None

from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

def files_in(dir):
    return list(sorted(Path(dir).glob('*')))


def save(img_tensor, file):
    if img_tensor.ndim == 4:
        assert len(img_tensor) == 1

    save_image(img_tensor, str(file))


def load(file):
    img = Image.open(str(file))
    img = img.convert('RGB')
    return img


def style_transforms(size=256):
    # any style input size is allowed
    return Compose([
        Resize(size=size),  # Resize to keep aspect ratio
        ToTensor()])


def content_transforms():
    # min_size is optional as content images have no size restrictions
    transforms = []
    transforms.append(ToTensor())
    return Compose(transforms)


class StylizationDataset(Dataset):
    def __init__(self, content_files, style_files, content_transform=None, style_transform=None):
        self.content_files = content_files
        self.style_files = style_files

        id = lambda x: x
        self.content_transform = id if content_transform is None else content_transform
        self.style_transform = id if style_transform is None else style_transform

    def __getitem__(self, idx):
        content_file, style_file = self.files_at_index(idx)

        content_img = load(content_file)
        style_img = load(style_file)

        content_img = self.content_transform(content_img)
        style_img = self.style_transform(style_img)

        return {
            'content': content_img,
            'style': style_img,
        }

    def __len__(self):
        return len(self.content_files) * len(self.style_files)

    def files_at_index(self, idx):
        content_idx = idx % len(self.content_files)
        style_idx = idx // len(self.content_files)

        assert 0 <= content_idx < len(self.content_files)
        assert 0 <= style_idx < len(self.style_files)
        return self.content_files[content_idx], self.style_files[style_idx]


class EndlessDataset(IterableDataset):
    """
    Wrapper for StylizationDataset which loops infinitely.
    Usefull when training based on iterations instead of epochs
    """

    def __init__(self, *args, **kwargs):
        self.dataset = StylizationDataset(*args, **kwargs)

    def __iter__(self):
        while True:
            idx = random.randrange(len(self.dataset))

            try:
                yield self.dataset[idx]
            except Exception as e:
                files = self.dataset.files_at_index(idx)
                warnings.warn(f'\n{str(e)}\n\tFiles: [{str(files[0])}, {str(files[1])}]')
