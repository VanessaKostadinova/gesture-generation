from random import Random

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, emb_model, h_size, n_layers, dropout=0.5, device="cpu"):
        super(Encoder, self).__init__()

        self.hidden_size = h_size
        self.num_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.emb_model = emb_model
        self.device = device

        embedding_weights = torch.tensor(self.emb_model.vectors, device=self.device)
        self.emb_size = embedding_weights.size()
        self.embedding = nn.Embedding.from_pretrained(embedding_weights)

        self.rnn = nn.LSTM(list(self.emb_size)[1], h_size, n_layers, dropout=dropout, batch_first=True)

    def forward(self, word_in, h_0=None):
        try:
            # try getting embedding
            embedding = torch.tensor(self.emb_model[word_in], device=self.device)
        except KeyError:
            # if vector unknown: set to zeroes
            embedding = torch.zeros(size=[list(self.emb_size)[1]], device=self.device)

        emb_in = self.dropout(embedding)

        if h_0 is None:
            out, h_out = self.rnn(emb_in.unsqueeze(0))
        else:
            out, h_out = self.rnn(emb_in.unsqueeze(0), h_0)

        return out, h_out


class Decoder(nn.Module):
    def __init__(self, num_of_joints, h_size, n_layers, dropout):
        super().__init__()

        self.hidden_dim = h_size
        self.n_layers = n_layers

        self.rnn = nn.LSTM(h_size, h_size, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(h_size, num_of_joints)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ctx_vec, h_0=None):
        if h_0 is None:
            out, h_out = self.rnn(ctx_vec.unsqueeze(0))
        else:
            h_0 = h_0.unsqueeze(0).reshape(2, 1, 2048)
            out, h_out = self.rnn(ctx_vec.unsqueeze(0), h_0)

        prediction = self.fc_out(out)

        return prediction, h_out


class AttnDecoder(nn.Module):
    def __init__(self, num_of_joints, h_size, n_layers, dropout=0.5, device="cpu"):
        super().__init__()

        self.hidden_dim = h_size
        self.n_layers = n_layers
        self.n_joints = num_of_joints
        self.device = device

        self.attn = nn.Linear(self.hidden_dim * 2, self.hidden_dim, device=self.device)
        self.attn_combine = nn.Linear(self.hidden_dim + self.n_joints, self.hidden_dim, device=self.device)
        self.rnn = nn.LSTM(self.hidden_dim, h_size, n_layers, dropout=dropout, batch_first=True, device=self.device)
        self.fc_out = nn.Linear(h_size, num_of_joints, device=self.device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, exp_in, h_0, enc_outs):
        h_pad = h_0[0][1].repeat(enc_outs.size()[0], 1)
        attn_weights = torch.tanh(self.attn(torch.cat([h_pad.unsqueeze(0), enc_outs.unsqueeze(0)], dim=2)))

        attn_weights = attn_weights.squeeze(0).transpose(0, 1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), enc_outs.unsqueeze(0))

        h_cat = torch.concat([exp_in, attn_applied[0][0]], dim=0)

        h_attn = self.attn_combine(h_cat.unsqueeze(0))
        relu = F.relu(h_attn)

        out, h_out = self.rnn(relu, h_0)
        out = self.fc_out(out)
        out = out.squeeze(0)

        return out, h_out


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, h_dim, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.h_dim = h_dim
        self.random = Random()

    def forward(self, transcript_list, expected_frames, n_joints, teacher_force_ratio=0.5):
        input_words = list(filter("<PAD>".__ne__, transcript_list))
        input_length = len(input_words)
        output_length = len(expected_frames)
        print(input_words)
        print(expected_frames.size())

        enc_os = torch.zeros(input_length, self.h_dim).to(self.device)
        dec_os = torch.zeros(output_length, n_joints).to(self.device)

        enc_o, enc_h = self.encoder(input_words[0][0])
        enc_os[0] = enc_o[0, 0]

        for index in range(1, len(input_words)):
            enc_o, enc_h = self.encoder(input_words[index][0], enc_h)
            enc_os[index] = enc_o[0, 0]

        dec_in = torch.tensor(expected_frames[0], device=self.device, dtype=torch.float)
        dec_o, dec_h = self.decoder(dec_in, enc_h, enc_os)
        dec_in = dec_o
        dec_os[0] = dec_o

        for index in range(1, len(expected_frames)):
            dec_o, dec_h = self.decoder(dec_in, dec_h, enc_os)
            dec_os[index] = dec_o
            best_guess = dec_o
            dec_in = torch.tensor(expected_frames[index],
                                  device=self.device,
                                  dtype=torch.float) if self.random.random() < teacher_force_ratio else best_guess
        dec_os = dec_os.squeeze(0)
        return dec_os

    def evaluate(self, transcript, length, n_joints):
        transcript = transcript.split(" ")
        input_length = len(transcript)
        enc_os = torch.zeros(input_length, self.h_dim).to(self.device)
        dec_os = torch.zeros(length, n_joints).to(self.device)

        enc_o, enc_h = self.encoder(transcript[0])
        enc_os[0] = enc_o[0, 0]

        for index in range(1, input_length):
            enc_o, enc_h = self.encoder(transcript[index], enc_h)
            enc_os[index] = enc_o[0, 0]

        dec_in = torch.zeros(n_joints, device=self.device, dtype=torch.float)
        dec_o, dec_h = self.decoder(dec_in, enc_h, enc_os)
        dec_in = dec_o

        for f in range(1, length):
            dec_o, dec_h = self.decoder(dec_in, dec_h, enc_os)
            dec_os[f] = dec_o
        dec_os = dec_os.squeeze(0)
        return dec_os
