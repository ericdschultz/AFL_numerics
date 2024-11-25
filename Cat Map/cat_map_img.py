"""
Performs the given cat map on an input image. Saves the output to the same path as the input file.

Command:
python cat_map_img.py file a b c d iters [-saveall] [-pixels PIXELS]

file - image file including extension
abcd - The cat map matrix elements, i.e. A = [[a,b],[c,d]]
iters - number of times to iterate the cat map
saveall (optional) - Include to save all intermediate cat map steps.
pixels (optional; default 2048) - The maximum pixel size to rescale the image to.
"""

import argparse
import numpy as np
from PIL import Image

def cat_map(A, data):
    """ Performs one iteration of the cat map A on data. Assumes image is square. """
    N = data.shape[0]
    out = np.empty_like(data)

    q,p = np.meshgrid(np.arange(N), np.arange(N), indexing='xy')
    qout = np.mod(A[0,0] * q + A[0,1] * p, N)
    pout = np.mod(A[1,0] * q + A[1,1] * p, N)

    # PIL indexes from the top, while p coordinate is from the bottom
    y = N - 1 - p
    yout = N - 1 - pout
    
    # Numpy arrays are yx-indexing
    out[yout, qout] = data[y, q]
    return out

def to_square(img, max_size):
    """ Resize image to be square. """
    width, height = img.size
    pixels = min(width, height, max_size)
    return img.resize((pixels, pixels), resample=Image.Resampling.LANCZOS)

def main(file, matrix, iters, saveall=False, pixels=2048):
    A = np.array(matrix).reshape((2,2))

    # Load and resize image
    img = Image.open(file, mode='r')
    sq_img = to_square(img, pixels)
    data = np.array(sq_img) # PIL indexes in x,y from the upper left
    img.close()

    # Save resized image and close
    ext = file.index('.')
    A_str = '-' + ''.join(map(str, matrix)) + '-cat'
    name = file[:ext] + A_str + '0' + file[ext:]
    sq_img.save(name)
    sq_img.close()

    # Cat map iterations
    for i in range(1, iters+1):
        data[:] = cat_map(A, data)
        if saveall or i == iters:
            newim = Image.fromarray(data)
            name = file[:ext] + A_str + str(i) + file[ext:]
            newim.save(name)
            newim.close()

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("file",
                        help="image file including extension")
    parser.add_argument("matrix", nargs=4, type=int,
                        help="matrix elements a b c d")
    parser.add_argument("iters", type=int,
                        help="iterations of the cat map")
    parser.add_argument("-saveall", action="store_true",
                        help="save all intermediate cat map steps")
    parser.add_argument("-pixels", type=int, default=2048,
                        help="maximum pixel size to rescale to (default 2048)")
    args = parser.parse_args()
    main(args.file, args.matrix, args.iters, saveall=args.saveall, pixels=args.pixels)
    
