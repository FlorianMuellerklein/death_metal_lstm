import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        # input embedding
        self.encoder = nn.Embedding(input_size, embed_size)
        # lstm weights
        self.weight_fh = nn.Linear(hidden_size, hidden_size)
        self.weight_ih = nn.Linear(hidden_size, hidden_size)
        self.weight_ch = nn.Linear(hidden_size, hidden_size)
        self.weight_oh = nn.Linear(hidden_size, hidden_size)
        self.weight_fx = nn.Linear(embed_size, hidden_size)
        self.weight_ix = nn.Linear(embed_size, hidden_size)
        self.weight_cx = nn.Linear(embed_size, hidden_size)
        self.weight_ox = nn.Linear(embed_size, hidden_size)
        # decoder
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, inp, h_0, c_0):
        # encode the input characters
        inp = self.encoder(inp)
        # forget gate
        f_g = F.sigmoid(self.weight_fx(inp) + self.weight_fh(h_0))
        # input gate
        i_g = F.sigmoid(self.weight_ix(inp) + self.weight_ih(h_0))
        # output gate
        o_g = F.sigmoid(self.weight_ox(inp) + self.weight_oh(h_0))
        # intermediate cell state
        c_tilda = F.tanh(self.weight_cx(inp) + self.weight_ch(h_0))
        # current cell state
        cx = f_g * c_0 + i_g * c_tilda
        # hidden state
        hx = o_g * F.tanh(cx)

        out = self.decoder(hx.view(1,-1))

        return out, hx, cx

    def init_hidden(self):
        h_0 = Variable(torch.zeros(1, self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(1, self.hidden_size)).cuda()
        return h_0, c_0

class miLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, output_size):
        super(miLSTM, self).__init__()

        self.hidden_size = hidden_size
        # input embedding
        self.encoder = nn.Embedding(input_size, embed_size)
        # lstm weights
        self.weight_fh = nn.Linear(hidden_size, hidden_size)
        self.weight_ih = nn.Linear(hidden_size, hidden_size)
        self.weight_zh = nn.Linear(hidden_size, hidden_size)
        self.weight_oh = nn.Linear(hidden_size, hidden_size)
        self.weight_fx = nn.Linear(embed_size, hidden_size)
        self.weight_ix = nn.Linear(embed_size, hidden_size)
        self.weight_zx = nn.Linear(embed_size, hidden_size)
        self.weight_ox = nn.Linear(embed_size, hidden_size)
        # alphas and betas
        self.alpha_f = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_f1 = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_f2 = nn.Parameter(torch.ones(1,hidden_size))
        
        self.alpha_i = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_i1 = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_i2 = nn.Parameter(torch.ones(1,hidden_size))
        
        self.alpha_o = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_o1 = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_o2 = nn.Parameter(torch.ones(1,hidden_size))
        
        self.alpha_z = nn.Parameter(torch.ones(1,hidden_size))
        self.alpha_z = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_z1 = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_z2 = nn.Parameter(torch.ones(1,hidden_size))
        # decoder
        self.decoder = nn.Linear(hidden_size, output_size)


    def forward(self, inp, h_0, c_0):
        # encode the input characters
        inp = self.encoder(inp)
        # forget gate
        f_g = F.sigmoid(self.alpha_f * self.weight_fx(inp) * self.weight_fh(h_0) +
                       (self.beta_f1 * self.weight_fx(inp)) + (self.beta_f2 * self.weight_fh(h_0)))
        # input gate
        i_g = F.sigmoid(self.alpha_i * self.weight_ix(inp) * self.weight_ih(h_0) +
                       (self.beta_i1 * self.weight_ix(inp)) + (self.beta_i2 * self.weight_ih(h_0)))
        # output gate
        o_g = F.sigmoid(self.alpha_o * self.weight_ox(inp) * self.weight_oh(h_0) +
                       (self.beta_o1 * self.weight_ox(inp)) + (self.beta_o2 * self.weight_oh(h_0)))
        # block input
        z_t = F.tanh(self.alpha_z * self.weight_zx(inp) * self.weight_zh(h_0) +
                    (self.beta_z1 * self.weight_zx(inp)) + (self.beta_z2 * self.weight_zh(h_0)))
        # current cell state
        cx = f_g * c_0 + i_g * z_t
        # hidden state
        hx = o_g * F.tanh(cx)

        out = self.decoder(hx.view(1,-1))

        return out, hx, cx

    def init_hidden(self):
        h_0 = Variable(torch.zeros(1, self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(1, self.hidden_size)).cuda()
        return h_0, c_0

'''
Same models but with layer normalization
'''

class LayerNormalization(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.hidden_size = hidden_size
        self.a2 = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z, dim=1)
        sigma = torch.std(z, dim=1)

        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a2 + self.b2
        return ln_out

class LN_miLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, output_size):
        super(LN_miLSTM, self).__init__()

        self.hidden_size = hidden_size
        # input embedding
        self.encoder = nn.Embedding(input_size, embed_size)
        # lstm weights
        self.weight_fh = nn.Linear(hidden_size, hidden_size)
        self.weight_ih = nn.Linear(hidden_size, hidden_size)
        self.weight_zh = nn.Linear(hidden_size, hidden_size)
        self.weight_oh = nn.Linear(hidden_size, hidden_size)
        self.weight_fx = nn.Linear(embed_size, hidden_size)
        self.weight_ix = nn.Linear(embed_size, hidden_size)
        self.weight_zx = nn.Linear(embed_size, hidden_size)
        self.weight_ox = nn.Linear(embed_size, hidden_size)
        # alphas and betas
        self.alpha_f = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_f1 = nn.Parameter(torch.zeros(1,hidden_size))
        self.beta_f2 = nn.Parameter(torch.zeros(1,hidden_size))

        self.alpha_i = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_i1 = nn.Parameter(torch.zeros(1,hidden_size))
        self.beta_i2 = nn.Parameter(torch.zeros(1,hidden_size))

        self.alpha_o = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_o1 = nn.Parameter(torch.zeros(1,hidden_size))
        self.beta_o2 = nn.Parameter(torch.zeros(1,hidden_size))

        self.alpha_z = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_z1 = nn.Parameter(torch.zeros(1,hidden_size))
        self.beta_z2 = nn.Parameter(torch.zeros(1,hidden_size))
        # decoder
        self.decoder = nn.Linear(hidden_size, output_size)
        # layer normalization
        self.lnx = LayerNormalization(hidden_size)
        self.lnh = LayerNormalization(hidden_size)
        self.lno = LayerNormalization(hidden_size)


    def forward(self, inp, h_0, c_0):
        # encode the input characters
        inp = self.encoder(inp)
        # forget gate
        f_g = F.sigmoid(self.alpha_f * self.lnx(self.weight_fx(inp)) * self.lnh(self.weight_fh(h_0)) +
                       (self.beta_f1 * self.lnx(self.weight_fx(inp))) + (self.beta_f2 * self.lnh(self.weight_fh(h_0))))
        # input gate
        i_g = F.sigmoid(self.alpha_i * self.lnx(self.weight_ix(inp)) * self.lnh(self.weight_ih(h_0)) +
                       (self.beta_i1 * self.lnx(self.weight_ix(inp))) + (self.beta_i2 * self.lnh(self.weight_ih(h_0))))
        # output gate
        o_g = F.sigmoid(self.alpha_o * self.lnx(self.weight_ox(inp)) * self.lnh(self.weight_oh(h_0)) +
                       (self.beta_o1 * self.lnx(self.weight_ox(inp))) + (self.beta_o2 * self.lnh(self.weight_oh(h_0))))
        # block input
        z_t = F.tanh(self.alpha_z * self.lnx(self.weight_zx(inp)) * self.lnh(self.weight_zh(h_0)) +
                    (self.beta_z1 * self.lnx(self.weight_zx(inp))) + (self.beta_z2 * self.lnh(self.weight_zh(h_0))))
        # current cell state
        cx = f_g * c_0 + i_g * z_t
        # hidden state
        hx = o_g * F.tanh(self.lno(cx))

        out = self.decoder(hx.view(1,-1))

        return out, hx, cx

    def init_hidden(self):
        h_0 = Variable(torch.zeros(1, self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(1, self.hidden_size)).cuda()
        return h_0, c_0
