# modeling libraries
import torch
import torch.nn as nn
import tokenizers
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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

# IMG_LENGTH = 64 # image side length 
TEXT_LENGTH = 128 # max text length
# TODO TODO TODO  # TODO TODO TODO  # TODO TODO TODO

BATCH_SIZE = 16
EPOCHS = 3

# create a tokenizer
class MuteEmbedsDataset(Dataset):

    # TODO TODO TODO
    def __init__(self,
                 text_url, photos_url,
                 truncate_text=128, vocab_size=None,
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
        self.seq_length = truncate_text

        # store the tokenizer
        self.tokenizer = tokenizer

        # if there are no vocab size, calculate it
        self.vocab_size = vocab_size if vocab_size else len(self.tokenizer)

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
        # to get some variation to break satble matchase, we adda random permutation
        return torch.tensor(self.images[max(0, idx+random.randint(-5,0))]) 

    def __get_text(self, idx):
        return self.tokenizer(self.lines[idx],
                              max_length=self.seq_length, truncation=True,
                              return_tensors='pt')

    def __getitem__(self, idx):
        # get the actual index to get from
        true_idx = idx if self.training else self.train_size+idx-1 # one-off error?

        # get a single sample of image AND text
        text_encoded = self.__get_text(true_idx)
        return {"text_sample": text_encoded["input_ids"][0],    # because we are encoding
                "text_mask": text_encoded["attention_mask"][0], # 1 sample at a time
                "image": self.__get_image(true_idx)}

# train and test dataloaders
def common_entries(*dcts):
    "https://stackoverflow.com/questions/16458340/python-equivalent-of-zip-for-dictionaries"
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)
        
def collate_and_pad(data, padding_idx=1):
    return {i[0]: pad_sequence(i[1:],
                               batch_first=True,
                               padding_value=(padding_idx if i[0] != "text_mask"
                                              else 0)) for i in common_entries(*data)}

train_dataset = MuteEmbedsDataset(TEXT_URL, PHOTOS_URL, truncate_text=TEXT_LENGTH)
train_loader = iter(DataLoader(train_dataset, collate_fn=collate_and_pad,
                               batch_size=BATCH_SIZE, shuffle=True))

test_dataset = MuteEmbedsDataset(TEXT_URL, PHOTOS_URL, training=False, truncate_text=TEXT_LENGTH)
test_loader = iter(DataLoader(test_dataset,
                              batch_size=BATCH_SIZE, shuffle=True))

# network!
class MuteEmbeds(nn.Module):

    def __init__(self, num_words, hidden_size=128):

        super(MuteEmbeds, self).__init__()

        # text emebding; PADDING FOR THIS TOKENIZER IS IDX=1
        self.embedding = nn.Embedding(num_words, hidden_size, padding_idx=1)

        # the encoder network
        encoder_layer = nn.TransformerEncoderLayer(d_model=encoding_size, )
        self.encoder = nn.TransformerEncoder(

        # the image decoder
        # self.img_dec = nn.Linear(

    def forward(self, x, embed=False):
        pass


