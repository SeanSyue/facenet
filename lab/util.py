import struct
from pathlib import Path
from itertools import chain

import numpy as np


def get_image_paths(dataset_dir):
    p_dataset = Path(dataset_dir)
    all_image_list = [x for x in chain(p_dataset.rglob('*.png'), p_dataset.rglob('*.jpg'))]
    return [str(x) for x in all_image_list]


def get_output_path(feature_out_path_, image_path_, lfw_dir_):
    relevant_img_path = Path(image_path_).relative_to(lfw_dir_)
    relevant_feat_filename = str(relevant_img_path.with_name(f"{str(relevant_img_path.name)}_facenet.bin"))
    output_path = Path(feature_out_path_).joinpath(relevant_feat_filename)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    return str(output_path)


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


def _write_npy(path, feat):
    np.save(path, feat)


def write_feat(path, feat, data_type):
    if data_type == 'IJBC':
        _write_npy(path, feat)
    elif data_type == 'MEGA':
        _write_mega(path, feat)
    else:
        raise ValueError('Unknown dataset type!')