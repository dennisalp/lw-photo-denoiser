import sys
import os
from pdb import set_trace as st
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar

from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model



################################################################
# Parameters
hh = 4000
ww = 6000
nn = 64
nc = 3
nb = 2**8
ne = 1

nh = hh//nn+1
nw = ww//nn+1
ph = (nh*nn-hh)//2
pw = (nw*nn-ww)//2

ks = 2



################################################################
# ML model
class Autoencoder(Model):
  
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(nn, nn, 1)),
      layers.Conv2D(32, (ks, ks), activation='relu', padding='valid', strides=2),
      layers.Conv2D(64, (ks, ks), activation='relu', padding='valid', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(64, kernel_size=ks, strides=2, activation='relu', padding='valid'),
      layers.Conv2DTranspose(32, kernel_size=ks, strides=2, activation='relu', padding='valid'),
      layers.Conv2D(1, kernel_size=(1, 1), strides=1, activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

ae = Autoencoder()
ae.compile(optimizer='adam', loss=losses.MeanSquaredError())



################################################################    
def pad_image(arr):
    return np.pad(arr, ((ph,ph), (pw,pw), (0,0)), 'constant', constant_values=np.median(arr))

    
def slice_image(arr):
    arr = arr.reshape((nn, nh, nn, nw, nc), order='F').swapaxes(1, 2)
    return arr.reshape((nn, nn, nh*nw*nc)).swapaxes(1,2).swapaxes(0,1)


def reassemble_images(slices):
    slices = slices.swapaxes(0,1).swapaxes(1,2)
    slices = slices.reshape((nn, nn, nh, nw, nc)).swapaxes(1, 2)
    return slices.reshape((nh*nn, nw*nn, nc), order='F')


def read_imgs():
    print('\n\n\nreading images')
    files = sorted(glob('dat/*.jpg'))[:2]
    nf = len(files)
    exif = []
    icc = []
    dpi = []
    tiles = np.empty((nh*nw*nc*nf, nn, nn), dtype=np.float32)
                          
    with alive_bar(len(files)) as bar: 
        for ii, ff in enumerate(files):
            with Image.open(ff) as img:
                exif.append(img.info['exif'])
                icc.append(img.info['icc_profile'])
                dpi.append(img.info['dpi'])
                arr = np.array(img)/nb
            
                arr = pad_image(arr) 
                tiles[ii*nh*nw*nc:(ii+1)*nh*nw*nc,:,:] = slice_image(arr)

            bar()

    print('\n\n')
    return {'files': files, 'exif': exif, 'icc': icc, 'dpi': dpi, 'tiles': tiles[..., tf.newaxis]}

    
def train(tiles):
    ae.fit(tiles, tiles, epochs=ne, shuffle=True)


def denoise(imgs):
    print('\n\n\ndenoising images')
    out = np.empty(imgs['tiles'].shape, dtype=np.float32)
    n = 1000
    i = imgs['tiles'].shape[0]//n+1
    
    with alive_bar(i) as bar:
        for ii in range(i):
            tiles = imgs['tiles'][ii*n:(ii+1)*n, :, :]
            out[ii*n:(ii+1)*n, :, :] = ae.decoder(ae.encoder(tiles)).numpy()
            bar()

    return out


def write_imgs(imgs):
    print('\n\n\nwriting images')
    f = imgs['files']
    with alive_bar(len(f)) as bar:
        for ii, ff in enumerate(f):
            tiles = imgs['out'][ii*nh*nw*nc:(ii+1)*nh*nw*nc, :, :]
            arr = reassemble_images(tiles)[ph:-ph, pw:-pw, :]
            arr = (255*arr).astype(np.uint8)

            exif = imgs['exif'][ii]
            icc = imgs['icc'][ii]
            dpi = imgs['dpi'][ii]
            op = ff.replace('dat', 'out')
            Image.fromarray(arr).save(op, exif=exif, icc_profile=icc, dpi=dpi, subsampling=0, quality=96, optimize=True)
            bar()
    print('\n\n')



################################################################
def main():
    imgs = read_imgs()
    train(imgs['tiles'])
    imgs['out'] = denoise(imgs)
    write_imgs(imgs)

    plt.imshow(imgs['tiles'][36,:,:]); plt.figure(); plt.imshow(imgs['out'][36,:,:]); plt.show()
    st()

if __name__ == '__main__':
    main()
