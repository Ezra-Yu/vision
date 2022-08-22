import io
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset


class PetrelFileClient():
    """Petrel storage backend (for internal use).
    Args:
        enable_mc (bool): whether to enable memcached support. Default: True.
    """

    def __init__(self, enable_mc=True):
        try:
            from petrel_client import client
        except ImportError:
            raise ImportError('Please install petrel_client to enable '
                              'PetrelBackend.')

        self._client = client.Client(enable_mc=enable_mc)

    def get(self, filepath):
        filepath = str(filepath)
        value = self._client.Get(filepath)
        value_buf = memoryview(value)
        return value_buf


FILE_CLIENT = {'petrel': PetrelFileClient}


class ClsDataset(Dataset):
    """Simple dataset for classification.
    Args:
        imgs_root (str): Image root of the dataset, should be s3 ceph path for
            spe. Can use local path for temporary test.
        annos_file (str): Annotation root of each example, should be local
            path which is pre-downloaded from ceph.
        transforms (torchvision.transforms.transforms.Compose): The data
            transformation pipeline that need to be applied in this task.
        classes (str): File path of the class name list.
        num_classes (int): Number of classes to be recognized.
        img_fc (str, optional): File client of loading image files. Should be
            `petrel` in spe. Can use `disk` for local test. Defaults to `disk`.
    """
    from mmcls.datasets.categories import IMAGENET_CATEGORIES
    CLASSES = IMAGENET_CATEGORIES

    def __init__(self, imgs_root, annos_file, transforms, img_fc='petrel'):
        self.imgs_root = imgs_root
        # self.annos_file = annos_file
        # load annotations from local path
        self.img_fc = FILE_CLIENT[img_fc]()
        self.data_info = self.load_annos(annos_file)
        self.transforms = transforms
        self.classes = self.CLASSES
        # self.label_dict = {k: v for v, k in enumerate(self.CLASSES)}
        # for anno in self.annos:
        #     self._parse_spe_anno(anno)

    def load_annos(self, annos_file):
        sample_bytes = self.img_fc.get(annos_file)
        if isinstance(sample_bytes, memoryview):
            sample_bytes = sample_bytes.tobytes()
        return sample_bytes.decode().strip().split('\n')

    def __getitem__(self, idx):
        """Get corresponding image and ground truth label for specific
        index."""
        img_filename, label = self.data_info[idx].split()
        img_bytes = self.img_fc.get(osp.join(self.imgs_root, img_filename))

        buff = io.BytesIO(img_bytes)
        img = Image.open(buff)
        img = img.convert("RGB")
        try:
            if self.transforms is not None:
                img = self.transforms(img)
        except RuntimeError as e:
            print(img_filename)
            print(img.size)
            print(img.mode)
            raise e
        return img, int(label)

    def __len__(self):
        return len(self.data_info)
        