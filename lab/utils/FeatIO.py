from pathlib import Path
from itertools import chain


def get_image_paths(dataset_dir):
    """ Get all image paths inside a dataset root """
    p_dataset = Path(dataset_dir)
    all_image_list = [x for x in chain(p_dataset.rglob('*.png'), p_dataset.rglob('*.jpg'))]
    return [str(x) for x in all_image_list]


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
