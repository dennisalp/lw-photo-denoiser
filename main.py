import sys
import os
from pdb import set_trace as st

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def main():
    with Image.open('dat/DSC02536.jpg') as img:
        exif = img.info['exif']
        icc = img.info['icc_profile']
        dpi = img.info['dpi']
        st()
        Image.fromarray(np.array(img)).save('dat/test.jpg', exif=exif, icc_profile=icc, dpi=dpi, subsampling=0, quality=96, optimize=True)

if __name__ == '__main__':
    main()
