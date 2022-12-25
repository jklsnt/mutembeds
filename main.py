# modeling libraries
import torch
import tokenizers
# NLTK friends
import nltk
from nltk import sent_tokenize

# nicies
from tqdm import tqdm

# stdlib utilities
import json
import glob

# look for the data files
gl = glob.glob("./data/cc_download_articles/**/*.json")

# process a sample
def process_sample(sample):
    sentences = [j for k in
                [sent_tokenize(i) for i in sample["maintext"].split("\n")]
                for j in k]

    return sentences

sentences = []

for sample in tqdm(gl):
    with open(sample, 'r') as df:
        try:
            sentences += process_sample(json.load(df))
        except json.JSONDecodeError:
            pass

with open("./data/cc_news.txt", 'w') as df:
    df.writelines([i+"\n" for i in sentences])



