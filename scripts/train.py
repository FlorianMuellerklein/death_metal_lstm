import sys
import math
import time
import string
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import miLSTM, LN_miLSTM, LSTM
from utils import list_iter, char2tensor

from sklearn.utils import shuffle

torch.cuda.set_device(1)

n_characters = len(string.printable)
alphabet = string.printable
n_epochs = 3
print_every = 500
plot_every = 100
hidden_size = 1024
lr = 0.001

# list of uppercase letters to init bandnames
uppers = string.ascii_uppercase

rnn_type = str(sys.argv[1])

if rnn_type == 'milstm':
    decoder = miLSTM(n_characters, hidden_size, 64, n_characters)
elif rnn_type == 'lstm':
    decoder = LSTM(n_characters, hidden_size, 64, n_characters)
else:
    decoder = LN_miLSTM(n_characters, hidden_size, 64, n_characters)

decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

decoder.cuda()

death_metal_bands = pd.read_csv('../data/death-metal/bands.csv')

band_raw = death_metal_bands['name'].tolist()

band_nms = []
for i, bnd in enumerate(band_raw):
    band_nms.append(bnd + '<EOS>')

print('Found', len(band_nms), 'bands!')

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(inp, target):
    decoder.train()
    hx, cx = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(len(inp)):
        output, hx, cx = decoder(inp[c], hx, cx)
        loss += criterion(output, target[c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / len(inp)

def evaluate(prime_str='A', predict_len=100, temperature=0.8):
    decoder.eval()


    hx, cx = decoder.init_hidden()
    prime_input = char2tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hx, cx = decoder(prime_input[p], hx, cx)
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hx, cx = decoder(inp, hx, cx)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = alphabet[top_i]
        predicted += predicted_char
        inp = char2tensor(predicted_char)

    return predicted.split('<EOS>')[0]

start = time.time()
all_losses = []
loss_avg = 0
chunk_counter = 0
try:
    for epoch in range(1, n_epochs + 1):
        band_nms = shuffle(band_nms)
        for inp, targs in list_iter(band_nms):
            loss = train(inp, targs)
            loss_avg += loss

            if chunk_counter % print_every == 0:
                print('Epoch:', epoch, 
                        'Loss:', np.round(loss_avg / print_every, decimals=3), 
                        '%:', np.round(chunk_counter / float(len(band_nms)), decimals=3) )

                rand_char = uppers[np.random.permutation(len(uppers))[0]]

                print(evaluate(rand_char, 100), '\n')

            if chunk_counter % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0

            chunk_counter += 1

        for param_group in decoder_optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.75
            print('New Learning rate:', param_group['lr'])

except KeyboardInterrupt:
    pass

torch.save(decoder.state_dict(), '../models/' + rnn_type + '_' + str(hidden_size) + '.pth')

with open('../plots/' + rnn_type + '_' + str(hidden_size) +'.pkl', 'wb') as f:
    pickle.dump(all_losses, f)

