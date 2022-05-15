import csv
import json
import os
from random import Random
import gensim
from gensim import downloader
import torch
import torch.optim
import torch.nn as nn
import re
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
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

        frame_duration = 0.04

        bvh_path = "F:\\fyp\\cut_data_openpose\\"
        transcript_path = "F:\\fyp\\transcript_aligned\\"

        for name in list_of_names:
            if os.path.exists(transcript_path + name + ".json") & os.path.exists(bvh_path + name + "\\ann_data.txt"):
                with open(transcript_path + name + ".json", "r") as json_file:
                    transcript_json = json.load(json_file)
                    if "words" in transcript_json.keys():
                        self.transcripts.append(transcript_json)
                    else:
                        continue
                with open(bvh_path + name + "\\gestures.json") as gesture_file:
                    gestures = json.load(gesture_file)

                    # for gesture in gestures:
                    #    s_time = gesture["t_0"][1] * frame_duration
                    #    p_time = gesture["p"][1] * frame_duration
                    #    e_time = gesture["t_1"][1] * frame_duration

                    # find which words match
                    #    print(transcript_json)
                    #    last_index = len(transcript_json["words"])
                    #    candidates = []

                    #    for word in transcript_json["words"]:
                    #        if "alignedWord" in word.keys():
                    #            start_time = word["start"]
                    #            end_time = word["end"]

                    #            if e:
                    #                candidates.append(word)

                with open(bvh_path + name + "\\ann_data_arm.txt", "r") as bvh_file:
                    bvh_rows = []
                    for line in bvh_file:
                        string_contents = line.split(",")
                        float_contents = []
                        for x in string_contents:
                            float_contents.append(float(x))
                        bvh_rows.append(float_contents)
                self.bvh_files.append(bvh_rows)

        # print(bvh_rows)
        # print(len(self.bvh_files))
        # self.bvh_files.append(bvh_rows)
        # lines = open(bvh_path + name + "\\ann_data.txt", "r").read().split(",")
        # for i in range(0, len(lines)):
        #    self.bvh_files.append(open(bvh_path + name + "\\ann_data.txt", "r").read().split(","))

    def __len__(self):
        return len(self.bvh_files)

    def __getitem__(self, index):
        # print(len(self.transcripts))
        # print(len(self.bvh_files))
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
    def __init__(self, emb_model, h_size, n_layers, dropout=0.5):
        super(Encoder, self).__init__()

        self.hidden_size = h_size
        self.num_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.emb_model = emb_model

        embedding_weights = torch.tensor(emb_model.vectors, device=device)
        self.emb_size = embedding_weights.size()

        self.embedding = nn.Embedding.from_pretrained(embedding_weights)

        self.rnn = nn.GRU(list(self.emb_size)[1], h_size, n_layers, dropout=dropout, batch_first=True)

    def forward(self, word_in, h_0=None):
        try:
            # try getting embedding
            norm = normalise_string(word_in)
            embedding = torch.tensor(embedding_model[norm], device=device)
        except KeyError:
            # if vector unknown: set to zeroes
            embedding = torch.zeros(size=[list(self.emb_size)[1]], device=device)

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

        self.rnn = nn.GRU(h_size, h_size, n_layers, dropout=dropout, batch_first=True)
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
    def __init__(self, num_of_joints, h_size, n_layers, dropout):
        super().__init__()

        self.hidden_dim = h_size
        self.n_layers = n_layers
        self.n_joints = num_of_joints

        self.attn = nn.Linear(self.hidden_dim * 2, self.hidden_dim, device=device)
        self.attn_combine = nn.Linear(self.hidden_dim + self.n_joints, self.hidden_dim, device=device)
        self.rnn = nn.GRU(self.hidden_dim, h_size, n_layers, dropout=dropout, batch_first=True, device=device)
        self.fc_out = nn.Linear(h_size, num_of_joints, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, exp_in, h_0, enc_outs):
        h_pad = h_0[1].repeat(enc_outs.size()[0], 1)
        # self.rnn.flatten_parameters()
        in_attn = torch.tanh(self.attn(torch.cat([h_pad.unsqueeze(0), enc_outs.unsqueeze(0)], dim=2)))

        attn_weights = F.softmax(in_attn, dim=1)
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
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, transcript_json, expected_frames, n_joints, teacher_force_ratio=0.5):
        # batch_size = output.shape[0]
        # print(transcript_input)
        input_words = transcript_json["words"]
        output_length = len(expected_frames)
        input_length = len(input_words)
        # print(output_length)

        predicted_frames = torch.zeros(output_length, n_joints).to(device)
        # print(input_words[0]["word"][0])
        enc_os = torch.zeros(input_length, hidden_size).to(device)

        enc_o, enc_h = self.encoder(input_words[0]["word"][0])
        enc_os[0] = enc_o[0, 0]

        for index in range(1, len(input_words)):
            enc_o, enc_h = self.encoder(input_words[index]["word"][0], enc_h)
            enc_os[index] = enc_o[0, 0]

        dec_in = torch.tensor(expected_frames[0], device=device, dtype=torch.float)
        predicted_frame, dec_h = self.decoder(dec_in, enc_h, enc_os)
        dec_in = predicted_frame
        predicted_frames[0] = predicted_frame

        for f in range(1, len(expected_frames)):
            # dec_in = torch.tensor(expected_frames[f], device=device, dtype=torch.float)
            predicted_frame, dec_h = self.decoder(dec_in, dec_h, enc_os)
            predicted_frames[f] = predicted_frame

            best_guess = predicted_frame
            dec_in = torch.tensor(expected_frames[f],
                                  device=device,
                                  dtype=torch.float) if random.random() < teacher_force_ratio else best_guess
        predicted_frames = predicted_frames.squeeze(0)
        return predicted_frames

    def evaluate(self, transcript, length, n_joints):
        transcript = transcript.split(" ")
        input_length = len(transcript)
        enc_os = torch.zeros(input_length, hidden_size).to(device)
        dec_os = torch.zeros(length, n_joints).to(device)

        enc_o, enc_h = self.encoder(transcript[0])
        enc_os[0] = enc_o[0, 0]

        for index in range(1, input_length):
            enc_o, enc_h = self.encoder(transcript[index], enc_h)
            enc_os[index] = enc_o[0, 0]

        dec_in = torch.zeros(n_joints, device=device, dtype=torch.float)
        dec_o, dec_h = self.decoder(dec_in, enc_h, enc_os)
        dec_in = dec_o

        for f in range(1, length):
            dec_o, dec_h = self.decoder(dec_in, dec_h, enc_os)
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
hidden_size = 256
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

step = 0

data = BVHDataset(os.listdir("F:\\fyp\\cut_data_openpose\\"))

train_iterator = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, pin_memory=True)

encoder = Encoder(embedding_model, hidden_size, num_layers, encoder_dropout).to(device)
decoder = AttnDecoder(num_joints, hidden_size, num_layers, decoder_dropout).to(device)

seq2seq = Seq2Seq(encoder, decoder).to(device).float()

#criterion = nn.MSELoss().to(device).float()

optimiser = optim.Adam(seq2seq.parameters(), lr=0.001, betas=(0.5, 0.8))

if load_model:
    seq2seq.load_state_dict(torch.load("./model.pt"))
    seq2seq.train()

epoch_losses = []
scaler = torch.cuda.amp.GradScaler()

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

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs}')
    epoch_loss = 0

    for i, (inputs, targets) in enumerate(train_iterator):
        optimiser.zero_grad()
        yhat = seq2seq.forward(inputs, targets, num_joints)

        with torch.cuda.amp.autocast():
            loss = custom_loss(yhat, torch.tensor(targets, dtype=torch.float, device=device)) #criterion(yhat, torch.tensor(targets, dtype=torch.float, device=device))
        scaler.scale(loss).backward()
        scaler.step(optimiser)

        scaler.update()
        epoch_loss += loss.item()

    torch.save(seq2seq.state_dict(), "../../models/model.pt")
    print(epoch_loss / len(train_iterator))
    epoch_losses.append(epoch_loss / len(train_iterator))

print(epoch_losses)
