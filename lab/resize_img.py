"""
DEPRECATED
Manually resize images in dataset root and save resized copies
"""
from functools import partial
from pathlib import Path
from multiprocessing import Pool
from PIL import Image


def main(output_path, im_path):
    print(im_path)
    im = Image.open(im_path)

    im.resize((160, 160), Image.BICUBIC).save(Path(output_path) / Path(im_path).name)


if __name__ == '__main__':

    im_root = Path('../../../IJB_release/IJBC/affine-112X112/affine')
    img_list = [n for n in [i for i in (Path(im_root).rglob(e) for e in ('*.png', '*.jpg')) for i in i]]
    output_path = '../../resize_out'

    func = partial(main, output_path)
    pool = Pool(30)
    pool.map(func, img_list)
    pool.close()
    pool.join()
