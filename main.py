# modeling libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import math
import random

# import pillow
from PIL import Image

TEXT_URL = "./data/cc_news.txt"
PHOTOS_URL = "./data/COCO.npy"

IMG_LENGTH = 64 # image side length 
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
                "text_mask": text_encoded["attention_mask"][0].float(), # 1 sample at a time
                "image": self.__get_image(true_idx).float()}

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

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

train_dataset = MuteEmbedsDataset(TEXT_URL, PHOTOS_URL, truncate_text=TEXT_LENGTH, tokenizer=tokenizer)
train_loader = iter(DataLoader(train_dataset, collate_fn=collate_and_pad,
                               batch_size=BATCH_SIZE, shuffle=True))

test_dataset = MuteEmbedsDataset(TEXT_URL, PHOTOS_URL, training=False, truncate_text=TEXT_LENGTH, tokenizer=tokenizer)
test_loader = iter(DataLoader(test_dataset, collate_fn=collate_and_pad,
                              batch_size=BATCH_SIZE, shuffle=True))

# network!
class MuteEmbeds(nn.Module):

    def __init__(self, vocab_size, image_length=IMG_LENGTH, max_text_length=TEXT_LENGTH, size=128):

        super(MuteEmbeds, self).__init__()

        # text emebding; PADDING FOR THIS TOKENIZER IS IDX=1
        self.text_embedding = nn.Embedding(vocab_size, size, padding_idx=1)
        self.image_preprocessing = nn.Linear(image_length, size)

        # the encoder network
        encoder_layer = nn.TransformerEncoderLayer(d_model=size, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # decoder
        self.text_decoder = nn.Linear(size, vocab_size)
        self.image_decoder = nn.Linear(size, image_length)

        # store size
        self.size = size
        self.max_text_length = max_text_length
        self.vocab_size = vocab_size

        # util layers
        self.sigmoid = nn.Sigmoid()
        self.cross_entropy = nn.CrossEntropyLoss()

    @staticmethod
    def positionalencoding1d(d_model, length_max):
        """
        PositionalEncoding2D: https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
        AttentionIsAllYouNeed: https://arxiv.org/pdf/1706.03762.pdf
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length_max, d_model)
        position = torch.arange(0, length_max).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

    def forward(self, x, mask=None):
        # depending on if text or not, the model behaves differently
        # in text, we are passed a series of embedding indicies,
        # so it will be of type int and have 2 dims. Otherwise, float
        # and 3 dims. So:
        text = (len(x.shape) == 2 and not torch.is_floating_point(x))

        if text:
            net = self.text_embedding(x)*math.sqrt(self.size) # TODO why?
        else:
            net = self.image_preprocessing(x)

        # sine wave positional encoding
        pos_encoding = self.positionalencoding1d(self.size, net.shape[1])
        net += pos_encoding

        net = self.encoder(net.transpose(0,1), src_key_padding_mask=mask).transpose(0,1) # transpose because its sequence first
                                                                                         # and then we transpose back

        if text:
            out = self.text_decoder(net)
        else:
            out = self.sigmoid(self.image_decoder(net))

        # apparently, not normalizing is the standard
        if text:
            loss = F.cross_entropy(out, F.one_hot(x, self.vocab_size).float())
        else:
            loss = torch.mean(torch.abs(out - x)) # we use MAE instead of SSE because of gradient EXPLOSURION from squaring

        return {
            "embedding": net,
            "out": out,
            "loss": loss
        }

sample = next(test_loader)
sample.keys()

me = MuteEmbeds(len(tokenizer))
sample["text_sample"].dtype
sample["image"].dtype
sample["image"].shape


sample['image']
me.image_preprocessing.weight.dtype
tmp1 = me(sample['image'])
tmp2 = me(sample['text_sample'], mask=sample['text_mask'].float())

tmp1["embedding"].shape
tmp2["embedding"].shape
tmp2["loss"]

