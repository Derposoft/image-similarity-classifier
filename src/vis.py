import tensorflow as tf
import keras
import pandas as pd

# Data prep
train_metadata = pd.read_csv('../input/shopee-product-matching/train.csv')

# phash distance calculation
def get_phash_dist(phash_a, phash_b):
    return phash_a - phash_b
def parse_phash(phash):
    return int(phash, 16)

# (highly primitive) 1-D segmentation with k-means
import sklearn as sk


train_metadata['image_phash'] = train_metadata['image_phash'].apply(parse_phash)
train_metadata['image_phash']

# SSIM and PSNR values?
#tf.image.ssim()

# network takes in both images

