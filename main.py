import sys
import os
from pdb import set_trace as st
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

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

nh = hh//nn+1
nw = ww//nn+1
ph = (nh*nn-hh)//2
pw = (nw*nn-ww)//2

ks = 3

################################################################
# ML model
class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(nn, nn, nc)),
      layers.Conv2D(16, (ks, ks), activation='relu', padding='same', strides=1),
      layers.Conv2D(8, (ks, ks), activation='relu', padding='same', strides=1)])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=ks, strides=1, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=ks, strides=1, activation='relu', padding='same'),
      layers.Conv2D(3, kernel_size=(ks, ks), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

ae = Autoencoder()
ae.compile(optimizer='adam', loss=losses.MeanSquaredError())



################################################################
def main():
    imgs = read_imgs()
    train(imgs['tiles'])
    imgs['out'] = denoise(imgs['tiles'])
    st()
    plt.imshow(imgs['tiles'][12,:,:,:]); plt.figure(); plt.imshow(imgs['out'][12,:,:,:]); plt.show()

    
def pad_image(arr):
    return np.pad(arr, ((ph,ph), (pw,pw), (0,0)), 'constant', constant_values=np.median(arr))

    
def slice_image(arr):
    arr = arr.reshape((nn, nh, nn, nw, nc), order='F').swapaxes(1, 2)
    return arr.reshape((nn, nn, nh*nw, nc)).swapaxes(1,2).swapaxes(0,1)


def reassemble_image():
    1


def read_imgs():
    print('reading images')
    files = sorted(glob('dat/*.jpg'))[:2]
    nf = len(files)
    exif = []
    icc = []
    dpi = []
    tiles = np.empty((nh*nw*nf, nn, nn, nc))
                          
    for ii, ff in enumerate(files):
        with Image.open(ff) as img:
            exif.append(img.info['exif'])
            icc.append(img.info['icc_profile'])
            dpi.append(img.info['dpi'])
            arr = np.array(img)/nb
            
        arr = pad_image(arr) 
        tiles[ii*nh*nw:(ii+1)*nh*nw,:,:,:] = slice_image(arr)
    return {'files': files, 'exif': exif, 'icc': icc, 'dpi': dpi, 'tiles': tiles[:1000,:,:,:]}

    
def train(tiles):
    ae.fit(tiles, tiles, epochs=6, shuffle=True)


def denoise(tiles):
    return ae.decoder(ae.encoder(tiles)).numpy()


def save_imgs():
    Image.fromarray(np.array(img)).save('dat/test.jpg', exif=exif, icc_profile=icc, dpi=dpi, subsampling=0, quality=96, optimize=True)



################################################################
if __name__ == '__main__':
    main()
