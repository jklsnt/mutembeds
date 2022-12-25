# modeling libraries
import torch
import torch.nn as nn
import tokenizers
import numpy as np

# Ling utilities
import nltk
from nltk import sent_tokenize
from transformers import AutoTokenizer


PHOTOS_GLOB = "./data/COCO/*.jpg"
# calculate image side length
SIDE_LENGTH = 64

# load raw texual data
with open(TEXT_URL, 'r') as df:
    lines = [i.strip() for i in df.readlines()]

# load raw image data
photos = glob.glob(PHOTOS_GLOB)

# process each image
def process_photo(photo):
    with Image.open(photo) as photo:
        photo_res = photo.resize([SIDE_LENGTH, SIDE_LENGTH]).convert("L")
        resized = np.array(photo_res)
        return resized

# process all images
photos_arr = []
for i in tqdm(photos):
    photos_arr.append(process_photo(i))
photos_arr = np.array(photos_arr)

