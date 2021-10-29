import glob
import os
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class DIBCO(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.data_list = self._get_files('data')
        self.target_list = self._get_files('target')

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.read_image(self.data_list[index], 'RGB')
        gt = self.read_image(self.target_list[index], 'RGB')

        # Apply data augmentation.
        if self.transform is not None:
            img = Image.fromarray(img)
            gt = Image.fromarray(gt)
            img, gt = self.transform(img, gt)

        return img, gt, self.data_list[index]

    def __len__(self) -> int:
        return len(self.data_list)

    def _get_files(self, data):
        """Gets files for the specified data type and dataset split.
            Args:
                data: String, desired data ('data' or 'target').
            Returns:
                A list of sorted file names or None when getting label for test set.
            """
        search_path = os.path.join(self.root,
                                   "gt" if data == "target" else "img", "*")
        filenames = glob.glob(search_path)
        return sorted(filenames)

    @staticmethod
    def read_image(file_name, format=None):
        image = Image.open(file_name)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image
