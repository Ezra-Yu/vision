'''Dataset for torchvision'''

import io

from mmcv.transforms import LoadImageFromFile as BaseLoadImageFromFile
from mmcls.datasets import ImageNet
from PIL import Image

__all__ = ['MMClsImageNet', 'create_dataset']

class MMClsImageNet(ImageNet):
    """用以替换 `torchvision.datasets`  的 `ImageNet` 和 `ImageFolder`.
    
    可以用一下代码实验。

    Example:

    >>> from dataset import MMClsImageNet
    >>> # 读本地的文件夹形式
    >>> train = MMClsImageNet("./data/imagenet/train") 
    >>> len(train)
    1281167
    >>> train[12131]
    (<PIL.Image.Image image mode=RGB size=500x333 at 0x7FDDAC778550>, 9)
    >>> 
    >>> # 读 ceph 上的标注格式 (s 集群上默认格式)
    >>> val = MMClsImageNet("./data/imagenet/val", ann_file="./data/imagenet/meta/val.txt", local=False, cluster_name ='openmmlab')
    >>> len(val)
    50000
    >>> val[1000]
    (<PIL.Image.Image image mode=RGB size=500x317 at 0x7FDDAC778430>, 188)
    """

    def __init__(self,
                 data_prefix,
                 ann_file="",
                 transforms = None,
                 local = True,
                 cluster_name ='openmmlab',
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
                                    './data/imagenet/': f'{cluster_name}:s3://openmmlab/datasets/classification/imagenet/',
                                    'data/imagenet/': f'{cluster_name}:s3://openmmlab/datasets/classification/imagenet/'
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
 

def create_dataset(
        name,
        root,
        ann_file="",
        local=True,
        cluster_name ='openmmlab',
        split='validation',
        search_split=True,
        class_map=None,
        load_bytes=False,
        is_training=False,
        download=False,
        batch_size=None,
        repeats=0,
        **kwargs
):
    """ 使timm.data.create_dataset支持 `MMClsImageNet`. 其本身保持timm的用法

    可以用一下代码实验。

    Example:

    >>> from dataset import create_dataset
    >>> # 读本地的文件夹形式
    >>> train = create_dataset('MMClsImageNet', './data/imagenet/train') 
    >>> len(train)
    114238
    >>> train[12131]
    (<PIL.Image.Image image mode=RGB size=500x333 at 0x7FDDAC778550>, 9)
    >>> 
    >>> # 读 ceph 上的标注格式 (s 集群上默认格式)
    >>> val = create_dataset("MMClsImageNet", "./data/imagenet/val", ann_file="./data/imagenet/meta/val.txt", local=False, cluster_name ='openmmlab')
    >>> len(val)
    50000
    >>> val[1000]
    (<PIL.Image.Image image mode=RGB size=500x317 at 0x7FDDAC778430>, 188)
    """
    name = name.lower()
    if name == 'mmclsimagenet':
        ds = MMClsImageNet(root, ann_file, local=local, cluster_name=cluster_name, **kwargs)
    else:
        try:
            import timm
        except:
            raise ImportError("Please install timm")
        ds = timm.data.create_dataset(name,
                                    root,
                                    split=split,
                                    search_split=search_split,
                                    class_map=class_map,
                                    load_bytes=load_bytes,
                                    is_training=is_training,
                                    download=download,
                                    batch_size=batch_size,
                                    repeats=repeats,
                                    **kwargs) 
    return ds


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
