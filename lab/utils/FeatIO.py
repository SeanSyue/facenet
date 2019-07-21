import struct
from pathlib import Path
from itertools import chain

import numpy as np


def get_image_paths(dataset_dir):
    """ Get all image paths inside a dataset root """
    p_dataset = Path(dataset_dir)
    all_image_list = [x for x in chain(p_dataset.rglob('*.png'), p_dataset.rglob('*.jpg'))]
    return [str(x) for x in all_image_list]


class FeatureWriter:
    def __init__(self, dataset_type):
        # Different dataset type is needed cause we need two different naming pattern
        dataset_type = dataset_type.upper()
        if dataset_type not in ['MEGA', 'IJBC']:
            raise ValueError("FeatureWriter only accept `dataset_type` argument as 'MEGA' or 'IJBC' only!")
        self.dataset_type = dataset_type

    def get_output_path(self, feature_out_path_, image_path_, lfw_dir_):
        """ Render output path for each feature file """
        relevant_img_path = Path(image_path_).relative_to(lfw_dir_)

        feat_name = ''
        if self.dataset_type == 'MEGA':
            feat_name = f"{str(relevant_img_path.name)}_facenet.bin"
        elif self.dataset_type == 'IJBC':
            feat_name = f"{str(relevant_img_path.stem)}.npy"

        output_path = Path(feature_out_path_) / relevant_img_path.with_name(feat_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return str(output_path)

    @staticmethod
    def _write_mega(path, m):
        cv_type_to_dtype = {
            5: np.dtype('float32'),
            6: np.dtype('float64')
        }
        dtype_to_cv_type = {v: k for k, v in cv_type_to_dtype.items()}

        """Write mat m to file f"""
        if len(m.shape) == 1:
            rows = m.shape[0]
            cols = 1
        else:
            rows, cols = m.shape

        with open(path, 'wb') as f:
            header = struct.pack('iiii', rows, cols, cols * 4, dtype_to_cv_type[m.dtype])
            f.write(header)
            f.write(m.data)

    @staticmethod
    def _write_npy(path, feat):
        np.save(path, feat)

    def write_feat(self, path, feat):
        """ Write feature file for different dataset type """
        if self.dataset_type == 'IJBC':
            self._write_npy(path, feat)
        elif self.dataset_type == 'MEGA':
            self._write_mega(path, feat)
