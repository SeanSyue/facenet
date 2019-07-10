from pathlib import Path
from itertools import chain

import numpy as np
from scipy import misc
import tensorflow as tf


def get_paths_ijbc(dataset_dir):
    all_image_list = [x for x in Path(dataset_dir).iterdir()]
    return [str(x) for x in all_image_list], [True] * len(all_image_list)


def get_image_paths(dataset_dir):
    p_dataset = Path(dataset_dir)
    all_image_list = [x for x in chain(p_dataset.rglob('*.png'), p_dataset.rglob('*.jpg'))]
    return [str(x) for x in all_image_list]
