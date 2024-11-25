"""
Performs the (r=1/2) baker's map on an input image. Saves the output to the same path as the input file.

Command:
python bakers_map_img.py file iters [-saveall] [-pixels PIXELS]

file - image file including path and extension
iters - number of times to iterate the baker's map
saveall (optional) - Include to save all intermediate baker's map steps
pixels (optional; default 2048) - The maximum pixel size to rescale the image to.
"""

import argparse
import numpy as np
from PIL import Image

def bakers_map(img):
    """ Performs one iteration of the baker's map on data. Assumes image is square. """
    out = np.empty_like(np.array(img))

    width, height = img.size
    # Other resamplers are faster, but do a poorer job.
    img2 = img.resize((width * 2, height // 2), resample=Image.Resampling.LANCZOS)
    data = np.array(img2)

    N = width
    # Top half
    # Note PIL indexes from the top left and Numpy arrays have yx-indexing
    out[:N//2,:] = data[:,N:]
    # Bottom half
    out[N//2:,:] = data[:,:N]

    img.close()
    img2.close()
    return Image.fromarray(out)

def to_square(img, max_size):
    """ Resize image to be square. """
    width, height = img.size
    pixels = min(width, height, max_size)
    if pixels % 2 == 1:
        pixels -= 1
    return img.resize((pixels, pixels), resample=Image.Resampling.LANCZOS)

def main(file, iters, saveall=False, pixels=2048):
    # Load and resize image
    img = Image.open(file, mode='r')
    newim = to_square(img, pixels)
    img.close()

    # Save resized image
    ext = file.index('.')
    name = file[:ext] + '-baker0' + file[ext:]
    newim.save(name)
    
    # Baker's map iterations
    for i in range(1, iters+1):
        newim = bakers_map(newim)
        if saveall or i == iters:
            name = file[:ext] + '-baker{}'.format(i) + file[ext:]
            newim.save(name)
    newim.close()

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("file",
                        help="image file including extension")
    parser.add_argument("iters", type=int,
                        help="iterations of the baker's map")
    parser.add_argument("-saveall", action="store_true",
                        help="save all intermediate baker's map steps")
    parser.add_argument("-pixels", type=int, default=2048,
                        help="maximum pixel size to rescale to (default 2048)")
    args = parser.parse_args()
    main(args.file, args.iters, saveall=args.saveall, pixels=args.pixels)
