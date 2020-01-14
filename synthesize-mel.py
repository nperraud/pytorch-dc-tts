#!/usr/bin/env python
"""Synthetize sentences into speech."""
__author__ = 'Erdene-Ochir Tuguldur'

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import os
import sys
from tqdm import *

import numpy as np
import torch

from models import Text2Mel
from hparams import HParams as hp
from utils import get_last_checkpoint_file_name, load_checkpoint, save_to_png
from datasets.lj_speech import vocab, get_test_data


SENTENCES = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "Large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds."
]

torch.set_grad_enabled(False)

text2mel = Text2Mel(vocab).eval()
last_checkpoint_file_name = get_last_checkpoint_file_name(os.path.join(hp.logdir, '%s-text2mel' % 'ljspeech'))
# last_checkpoint_file_name = 'logdir/%s-text2mel/step-300K.pth' % 'ljspeech'
if last_checkpoint_file_name:
    print("loading text2mel checkpoint '%s'..." % last_checkpoint_file_name)
    load_checkpoint(last_checkpoint_file_name, text2mel, None)
else:
    print("text2mel not exits")
    sys.exit(1)

os.makedirs('samples-mel', exist_ok=True)
mels = []
# synthetize by one by one because there is a batch processing bug!
for i in range(len(SENTENCES)):
    sentences = [SENTENCES[i]]

    max_N = len(SENTENCES[i])
    L = torch.from_numpy(get_test_data(sentences, max_N))
    zeros = torch.from_numpy(np.zeros((1, hp.n_mels, 1), np.float32))
    Y = zeros
    A = None

    for t in tqdm(range(hp.max_T)):
        _, Y_t, A = text2mel(L, Y, monotonic_attention=True)
        Y = torch.cat((zeros, Y_t), -1)
        _, attention = torch.max(A[0, :, -1], 0)
        attention = attention.item()
        if L[0, attention] == vocab.index('E'):  # EOS
            break


    Y = Y.cpu().detach().numpy()
    A = A.cpu().detach().numpy()

    save_to_png('samples-mel/%d-att.png' % (i + 1), A[0, :, :])
    save_to_png('samples-mel/%d-mel.png' % (i + 1), Y[0, :, :])
#     mels.append(Y[0, :, :])
    
    np.savez('samples-mel/sentence-{}.npz'.format(i+1), mel=Y, sentence=sentences[0])
