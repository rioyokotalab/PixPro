import io
import logging
import os
import time
import random
import warnings

import torch.distributed as dist
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO

from .zipreader import is_zip_path, ZipReader


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions, is_bdd100k=False, n_frames=1):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        videos = []

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    if is_bdd100k:
                        videos.append(item)
                        continue
                    images.append(item)
        if is_bdd100k:
            images.append(videos)

    if is_bdd100k:
        images = VideoSample(images, n_frames=n_frames)

    return images


def make_dataset_with_ann(ann_file, img_prefix, extensions, dataset='ImageNet', n_frames=1):
    images = []

    # make COCO dataset
    if dataset == 'COCO':
        coco = COCO(ann_file)
        img_ids = coco.getImgIds()
        for idx in img_ids:
            im_file_name = coco.loadImgs([idx])[0]['file_name']
            class_index = 0

            assert str.lower(os.path.splitext(im_file_name)[-1]) in extensions
            item = (os.path.join(img_prefix, im_file_name), class_index)

            images.append(item)

        return images

    is_bdd100k = dataset == "bdd100k"
    pre_video_id = 0

    # make ImageNet or VOC dataset
    with open(ann_file, "r") as f:
        contents = f.readlines()
        videos = []
        for line_str in contents:
            path_contents = [c for c in line_str.split('\t')]
            im_file_name = path_contents[0]
            class_index = int(path_contents[1])

            assert str.lower(os.path.splitext(im_file_name)[-1]) in extensions
            item = (os.path.join(img_prefix, im_file_name), class_index)

            if is_bdd100k:
                if pre_video_id != class_index:
                    if len(videos) > 0:
                        images.append(videos)
                    videos = []
                    pre_video_id = class_index
                videos.append(item)
                continue

            images.append(item)

    if is_bdd100k:
        if len(videos) > 0:
            images.append(videos)
        images = VideoSample(images, n_frames=n_frames)

    return images


class VideoSample(data.Dataset):
    def __init__(self, samples, n_frames=1):
        super().__init__()
        self.samples = samples
        self.n_frames = n_frames

    def __getitem__(self, index):
        video = self.samples[index]
        n_video = len(video)
        n_frames = self.n_frames if n_video >= self.n_frames else n_video
        len_img = n_video - n_frames
        local_i = random.randint(0, len_img)
        path, target = video[local_i]

        if self.n_frames > 1:
            if n_frames <= 1:
                video_name = os.path.dirname(path)
                warnings.warn(f"{n_frames} videos can only be loaded in {video_name}")
            next_local_i = local_i + n_frames - 1
            next_path, next_target = video[next_local_i]
            path = [path, next_path]
            target = [target, next_target]

        return path, target

    def __len__(self):
        return len(self.samples)


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, ann_file='', img_prefix='', transform=None, target_transform=None,
                 cache_mode="no", dataset='ImageNet', n_frames=1):
        # image folder mode
        if ann_file == '':
            _, class_to_idx = find_classes(root)
            samples = make_dataset(root, class_to_idx, extensions, dataset == "bdd100k", n_frames)
        # zip mode
        else:
            samples = make_dataset_with_ann(os.path.join(root, ann_file),
                                            os.path.join(root, img_prefix),
                                            extensions,
                                            dataset,
                                            n_frames)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.labels = [y_1k if not isinstance(y_1k, list) else y_1k[0] for _, y_1k in samples]
        self.classes = list(set(self.labels))

        self.transform = transform
        self.target_transform = target_transform

        self.cache_mode = cache_mode
        if self.cache_mode != "no":
            self.init_cache()

    def init_cache(self):
        assert self.cache_mode in ["part", "full"]
        n_sample = len(self.samples)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        samples_bytes = [None for _ in range(n_sample)]
        start_time = time.time()
        for index in range(n_sample):
            if index % (n_sample//10) == 0:
                t = time.time() - start_time
                logger = logging.getLogger(__name__)
                logger.info(f'cached {index}/{n_sample} takes {t:.2f}s per block')
                start_time = time.time()
            path, target = self.samples[index]
            if self.cache_mode == "full" or index % world_size == global_rank:
                samples_bytes[index] = (ZipReader.read(path), target)
            else:
                samples_bytes[index] = (path, target)
        self.samples = samples_bytes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if isinstance(sample, list):
            sample = sample[0]
            target = target[0]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if isinstance(path, bytes):
        img = Image.open(io.BytesIO(path))
    elif is_zip_path(path):
        data = ZipReader.read(path)
        img = Image.open(io.BytesIO(data))
    else:
        # with open(path, 'rb') as f:
        #     img = Image.open(f)
        img = Image.open(path)
    return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_img_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def default_imgs_loader(path_list):
    sample = []
    for path in path_list:
        sample.append(default_img_loader(path))
    return sample



class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, ann_file='', img_prefix='', transform=None, target_transform=None,
                 loader=default_img_loader, cache_mode="no", dataset='ImageNet',
                 two_crop=False, return_coord=False, n_frames=1):
        if n_frames > 1 and dataset == "bdd100k":
            loader = default_imgs_loader
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          ann_file=ann_file, img_prefix=img_prefix,
                                          transform=transform, target_transform=target_transform,
                                          cache_mode=cache_mode, dataset=dataset, n_frames=n_frames)
        self.imgs = self.samples
        self.two_crop = two_crop
        self.return_coord = return_coord
        self.same_two = False
        # if transform is not None:
        #     if isinstance(self.transform, tuple) and len(self.transform) == 2:
        #         self.same_two = transform[0].same_two
        #     else:
        #         self.same_two = transform.same_two

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        images = self.loader(path)
        if not isinstance(images, list):
            images = [images]
        if isinstance(target, list):
            target = target[0]
        coord = None

        if self.transform is not None:
            if isinstance(self.transform, tuple) and len(self.transform) == 2:
                img = self.transform[0](images[0], coord=coord)
            else:
                img = self.transform(images[0], coord=coord)
        else:
            img = images[0]

        if isinstance(img, tuple):
            tmp_img, coord = img
            self.same_two = isinstance(coord, list)
            if self.same_two:
                img = tmp_img

        if self.target_transform is not None:
            target = self.target_transform(target, coord=coord)

        if self.two_crop:
            if isinstance(self.transform, tuple) and len(self.transform) == 2:
                img2 = self.transform[1](images[-1], coord=coord)
            else:
                img2 = self.transform(images[-1], coord=coord)

        if self.same_two:
            if coord is not None:
                coord, _ = coord
            img = (img, coord)
            if self.two_crop:
                img2, coord2 = img2
                if coord2 is not None:
                    coord2, _ = coord2
                img2 = (img2, coord2)

        if self.return_coord:
            assert isinstance(img, tuple)
            img, coord = img

            if self.two_crop:
                img2, coord2 = img2
                return img, img2, coord, coord2, index, target
            else:
                return img, coord, index, target
        else:
            if isinstance(img, tuple):
                img, coord = img

            if self.two_crop:
                if isinstance(img2, tuple):
                    img2, coord2 = img2
                return img, img2, index, target
            else:
                return img, index, target
