# modeling libraries
import torch
import torch.nn as nn
import tokenizers
import numpy as np

from torch.utils.data import Dataset, DataLoader

# Ling utilities
import nltk
from nltk import sent_tokenize
from transformers import AutoTokenizer

# nicies
from tqdm import tqdm

# stdlib utilities
import json
import glob
import random

# import pillow
from PIL import Image

TEXT_URL = "./data/cc_news.txt"
PHOTOS_URL = "./data/COCO.npy"

SEQ_LENGTH = 64 # BOTH image side length and max sentence length
# TODO TODO TODO  # TODO TODO TODO  # TODO TODO TODO

BATCH_SIZE = 16
EPOCHS = 3

# create a tokenizer
class MuteEmbedsDataset(Dataset):

    # TODO TODO TODO
    def __init__(self,
                 text_url, photos_url,
                 seq_length=64,
                 # train/test vars
                 training=True, test_split=0.1,
                 # PADDING FOR THIS TOKENIZER IS IDX=1
                 tokenizer=AutoTokenizer.from_pretrained("facebook/bart-large")):

        # init
        super(MuteEmbedsDataset, self).__init__()

        # load raw texual data
        with open(text_url, 'r') as df:
            self.lines = [i.strip() for i in df.readlines()]

        # load image data
        self.images = np.load(photos_url)/255 # norm to 255

        # get image size
        # TODO we assume square images
        self.seq_length = seq_length

        # store the tokenizer
        self.tokenizer = tokenizer

        # training storage
        self.training = training

        # get sizes
        raw_size = min(len(self.lines), len(self.images))
        self.train_size = int(raw_size*(1-test_split))
        self.test_size = int(raw_size*test_split)

    def __len__(self):
        # given train/test, diff. lengnths exist
        return self.train_size if self.training else self.test_size

    def __get_image(self, idx):
        return torch.tensor(self.images[idx])

    def __get_text(self, idx):
        return self.tokenizer(self.lines[idx],
                              max_length=self.seq_length, truncation=True,
                              padding="max_length", return_tensors='pt')["input_ids"]

    def __getitem__(self, idx):
        # get the actual index to get from
        true_idx = idx if self.training else self.train_size+idx-1 # one-off error?

        # get a single sample of image AND text
        return (self.__get_image(true_idx),
                self.__get_text(true_idx))

# train and test dataloaders
train_loader = iter(DataLoader(MuteEmbedsDataset(TEXT_URL, PHOTOS_URL),
                               batch_size=BATCH_SIZE, shuffle=True))
test_loader = iter(DataLoader(MuteEmbedsDataset(TEXT_URL, PHOTOS_URL, training=False),
                              batch_size=BATCH_SIZE, shuffle=True))

# initial

