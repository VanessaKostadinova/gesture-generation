import gensim
from gensim import downloader
import torch.optim
from pytorch.model.seq2seq_model import Encoder, AttnDecoder, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = gensim.downloader.load('glove-wiki-gigaword-300')

num_joints = 22
hidden_size = 512
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

transcript = "I have ten words here and they have varying length"
duration = 128

encoder = Encoder(embedding_model, hidden_size, num_layers, encoder_dropout, device).to(device)
decoder = AttnDecoder(num_joints, hidden_size, num_layers, decoder_dropout, device).to(device)

seq2seq = Seq2Seq(encoder, decoder, hidden_size, device).to(device).float()

seq2seq.load_state_dict(torch.load("./model_lstm_arm1.pt"))
seq2seq.eval()
print("pre-eval")
out = seq2seq.evaluate(transcript, duration, num_joints)
print("post-eval")
out_path = "../test.csv"

with open(out_path, "w") as file:
    out_list = out.tolist()
    for i in range(0, len(out_list)):
        row = out_list[i]
        out_strings = [str(round(x, 1)) for x in row]
        for k in range(0, len(out_strings)):
            file.write(out_strings[k])
            if k != len(out_strings) - 1:
                file.write(" ")

        if i != len(out_list) - 1:
            file.write("\n")
