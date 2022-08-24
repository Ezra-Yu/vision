'''Dataset for torchvision'''

import io

from mmcv.transforms import LoadImageFromFile as BaseLoadImageFromFile
from mmcls.datasets import ImageNet
from PIL import Image

class PILLoadImageFromFile(BaseLoadImageFromFile):

    def transform(self, results: dict):
        filename = results['img_path']
        try:
            img_bytes = self.file_client.get(filename)
            buff = io.BytesIO(img_bytes)
            results['img'] = Image.open(buff)
            
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        results['img_shape'] = results['img'].size
        results['ori_shape'] = results['img'].size
        return results


class ClsDataset(ImageNet):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    The dataset supports two kinds of annotation format. More details can be
    found in :class:`CustomDataset`.
    
    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        **kwargs: Other keyword arguments in :class:`CustomDataset` and
            :class:`BaseDataset`.
    """  # noqa: E501

    def __init__(self,
                 data_prefix,
                 ann_file,
                 transforms = None,
                 local = True,
                 **kwargs):
        super().__init__(
            ann_file=ann_file,
            data_prefix=data_prefix,
            **kwargs)
        self.transforms = transforms
        if local == True:
            file_client_args: dict = dict(backend='disk')
            self.load = PILLoadImageFromFile(file_client_args=file_client_args)
        else:
            file_client_args = dict(
                                backend='petrel',
                                path_mapping=dict({
                                    '.data/imagenet/': 's3://openmmlab/datasets/classification/imagenet/',
                                    'data/imagenet/': 's3://openmmlab/datasets/classification/imagenet/'
                                }))
            self.load = PILLoadImageFromFile(file_client_args=file_client_args)

    def __getitem__(self, idx: int) -> dict:
        result = super().__getitem__(idx)
        result = self.load(result)
        img, label = result['img'], result['gt_label']
        img = img.convert("RGB")
        try:
            if self.transforms is not None:
                img = self.transforms(img)
        except RuntimeError as e:
            print(result)
            raise e
        return img, int(label)     
 