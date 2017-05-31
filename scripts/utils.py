import json
import glob
import string

import torch
from torch.autograd import Variable

alphabet = string.printable

def char2tensor(string):
    '''
    Taken from https://github.com/spro/practical-pytorch/tree/master/char-rnn-generation
    Assigns a character index to each character
    '''
    tensor = torch.zeros(len(string)).cuda().long()
    for i, c in enumerate(string):
        try:
        	tensor[i] = alphabet.index(c)
        except ValueError:
        	continue
    return Variable(tensor)

def list_iter(band_list):
    '''
    Process the text squentially
    Return offset letter indices
    '''
    for band in band_list:
        # convert the tweet into idicies
        char_tensor = char2tensor(band)
        inp = char_tensor[:-1] # all but last
        targ = char_tensor[1:] # offset by one
        
        yield inp, targ

def tweet_loader():
	tweets = []
	file_list = glob.glob('../data/*training*/*.json')
	for json_fl in file_list:
		with open(json_fl) as f:
			twts = json.load(f)

		for i, twt in enumerate(twts):
			tweets.append(twt['text'] + '<EOS>')

	return tweets