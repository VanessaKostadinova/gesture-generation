import csv
import json
import os
from random import Random
import gensim
from gensim import downloader
from gensim.utils import tokenize
import torch
import torch.optim
import torch.nn as nn
import re
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchtext.data import get_tokenizer
from torch.utils.tensorboard import SummaryWriter

# out_file = open("./model", "x")

# out_file.write(gensim.downloader.load('glove-wiki-gigaword-300'))
# weights = torch.FloatTensor(model.vectors)
# embedding_weights = torch.FloatTensor(model.vectors)
# print(embedding_weights)
# print(list(embedding_weights.size())[1])

random = Random()


class BVHDataset(Dataset):
    def __init__(self, list_of_names):
        self.transcripts = []
        self.bvh_files = []

        bvh_path = "F:\\fyp\\cut_data_openpose\\"
        transcript_path = "F:\\fyp\\transcript_aligned\\"

        for name in list_of_names:
            if os.path.exists(transcript_path + name + ".json") & os.path.exists(bvh_path + name + "\\ann_data_arm.txt"):
                with open(transcript_path + name + ".json", "r") as json_file:
                    transcript_json = json.load(json_file)
                    if "words" in transcript_json.keys():
                        transcript_words = [normalise_string(e["word"]) for e in transcript_json["words"]]
                        self.transcripts.append(transcript_words)
                    else:
                        continue
                with open(bvh_path + name + "\\ann_data_arm.txt", "r") as bvh_file:
                    bvh_rows = []
                    for line in bvh_file:
                        string_contents = line.split(",")
                        float_contents = []
                        for x in string_contents:
                            float_contents.append(float(x))
                        bvh_rows.append(float_contents)
                    if len(bvh_rows) == 0:
                        self.transcripts.remove(transcript_json)
                    else:
                        self.bvh_files.append(bvh_rows)

    def __len__(self):
        return len(self.bvh_files)

    def __getitem__(self, index):
        return [self.transcripts[index], self.bvh_files[index]]


def prepare_embeddings():
    index = 0
    word2idx = {}
    with open("C:\\Users\\vanes\\gensim-data\\word2vec-google-news-300\\GoogleNews-vectors-negative300.bin",
              'rb') as file:
        for line in file:
            contents = line.decode().split()
            word2idx[contents[0]] = index
    return word2idx


def normalise_string(string):
    string = string.lower()
    string = re.sub(r"[\d+]", "", string)
    string = re.sub(r"[^\w\s]", "", string)
    string = string.strip()
    return string


class Encoder(nn.Module):
    def __init__(self, emb_model, h_dim, n_layers, dropout=0.5):
        super(Encoder, self).__init__()

        self.num_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.emb_model = emb_model

        embedding_weights = torch.tensor(emb_model.vectors, device=device)
        self.emb_size = list(embedding_weights.size())[1]

        self.rnn = nn.LSTM(self.emb_size, h_dim, n_layers, dropout=dropout, batch_first=True)

    def forward(self, word_in, h_0=None):
        try:
            # try getting embedding
            embedding = torch.tensor(self.emb_model[word_in], device=device)
        except KeyError:
            # if vector unknown: set to zeroes
            embedding = torch.zeros(size=self.emb_size, device=device)

        emb_in = self.dropout(embedding)

        if h_0 is None:
            out, h_out = self.rnn(emb_in.unsqueeze(0))
        else:
            print(h_0[0].size())
            print(emb_in.unsqueeze(0).size())
            out, h_out = self.rnn(emb_in.unsqueeze(0), h_0)

        return out, h_out


class Decoder(nn.Module):
    def __init__(self, n_joints, h_dim, n_layers, dropout):
        super().__init__()

        self.rnn = nn.LSTM(n_joints, h_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(h_dim, n_joints)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_o, h_0):
        print(h_0[0].size())
        print(dec_o.unsqueeze(0).size())
        out, h_out = self.rnn(dec_o.unsqueeze(0), h_0)

        prediction = self.fc_out(out)

        return prediction, h_out

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, transcript_list, expected_frames, n_joints, teacher_force_ratio=0.5):
        input_words = transcript_list
        input_length = len(input_words)
        output_length = len(expected_frames)

        # empty struct for storing outputs
        dec_os = torch.zeros(output_length, n_joints).to(device)

        # set up encoder
        enc_o, enc_h = self.encoder(input_words[0][0])

        for index in range(1, input_length):
            enc_o, enc_h = self.encoder(input_words[index][0], enc_h)

        # set up decoder
        dec_in = torch.tensor(expected_frames[0], device=device, dtype=torch.float)
        dec_o, dec_h = self.decoder(dec_in, enc_h)
        dec_in = dec_o
        # dec_os = decoder outputs/predicted frames
        dec_os[0] = dec_o

        # for each output, run decoder
        for f in range(1, output_length):
            dec_o, dec_h = self.decoder(dec_in, dec_h)
            dec_os[f] = dec_o

            dec_in = torch.tensor(expected_frames[f],
                                  device=device,
                                  dtype=torch.float) if random.random() < teacher_force_ratio else dec_o

        dec_os = dec_os.squeeze(0)
        return dec_os

    def evaluate(self, transcript, length, n_joints):
        input_length = len(transcript)

        # empty struct for storing outputs
        dec_os = torch.zeros(length, n_joints).to(device)

        enc_o, enc_h = self.encoder(transcript[0])

        for index in range(1, input_length):
            enc_o, enc_h = self.encoder(transcript[index], enc_h)

        dec_in = torch.zeros(n_joints, device=device, dtype=torch.float)
        dec_o, dec_h = self.decoder(dec_in, enc_h)
        dec_in = dec_o

        for f in range(1, length):
            dec_o, dec_h = self.decoder(dec_in, dec_h)
            dec_os[f] = dec_o
        dec_os = dec_os.squeeze(0)
        return dec_os

# vocab = set(test_sentence)
# word_to_ix = {word: i for i, word in enumerate(vocab)}

num_epochs = 20
learning_rate = 0.001
batch_size = 1

load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = gensim.downloader.load('glove-wiki-gigaword-300')

num_joints = 22
hidden_size = 512
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

data = BVHDataset(os.listdir("F:\\fyp\\cut_data_openpose\\"))

train_iterator = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, pin_memory=True)

encoder = Encoder(embedding_model, hidden_size, num_layers, encoder_dropout).to(device)
decoder = Decoder(num_joints, hidden_size, num_layers, decoder_dropout).to(device)

seq2seq = Seq2Seq(encoder, decoder).to(device).float()

optimiser = optim.Adam(seq2seq.parameters(), lr=0.001, betas=(0.5, 0.8))

#seq2seq.load_state_dict(torch.load("./model_lstm_arm.pt"))
#seq2seq.eval()
#print("hi")
#out = seq2seq.evaluate("i", 2, num_joints)

#with open("./test.csv", "w") as file:
#    out_list = out.tolist()
#    for i in range (0, len(out_list)):
#        row = out_list[i]
#        out_strings = [str(round(x, 1)) for x in row]
#        for k in range (0, len(out_strings)):
#            file.write(out_strings[k])
#            if k != len(out_strings) - 1:
#                file.write(" ")
#
#        if i != len(out_list) - 1:
#            file.write("\n")

loss_i = 0

def custom_loss(output, target):
    n_element = output.numel()

    # MSE
    l1_loss = F.l1_loss(output, target)
    l1_loss *= 5

    # continuous motion
    diff = [abs(output[:, n] - output[:, n-1]) for n in range(1, output.shape[1])]
    cont_loss = torch.sum(torch.stack(diff)) / n_element
    cont_loss *= 0.1

    # motion variance
    norm = torch.norm(output, 2, 1)
    var_loss = -torch.sum(norm) / n_element
    var_loss *= 0.5

    l = l1_loss + cont_loss + var_loss

    # inspect loss terms
    global loss_i
    if loss_i == 100:
        loss_i = 0
    loss_i += 1

    return l

if load_model:
    seq2seq.load_state_dict(torch.load("model_name"))
    seq2seq.train()

epoch_losses = []
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs}')
    epoch_loss = 0

    for i, (inputs, targets) in enumerate(train_iterator):
        optimiser.zero_grad()
        yhat = seq2seq.forward(inputs, targets, num_joints)

        with torch.cuda.amp.autocast():
            loss = custom_loss(yhat, torch.tensor(targets, dtype=torch.float, device=device))

        scaler.scale(loss).backward()
        scaler.step(optimiser)

        scaler.update()
        epoch_loss += loss.item()

    print(epoch_loss / len(train_iterator))
    epoch_losses.append(epoch_loss / len(train_iterator))

print(epoch_losses)
