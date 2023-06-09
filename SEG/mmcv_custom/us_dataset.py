# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class USDataset(CustomDataset):
    """DFUC dataset.

    In segmentation map annotation for DFUC2022, 0 stands for background,
    which is included in 2 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    CLASSES = ('background', 'nodule')

    PALETTE = [[0,0,0], [1,1,1]]

    def __init__(self, **kwargs):
        super(USDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        # import pdb;pdb.set_trace()
        assert self.file_client.exists(self.img_dir)