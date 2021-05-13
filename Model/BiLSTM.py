import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=True, bidirectional=True)

    def forward(self, inputs, sent_lens):
        # inputs = (batch,seq_len, embedding_dim)
        
        packed_embedded = pack_padded_sequence(inputs, sent_lens, batch_first=True, enforce_sorted=False)
        # self.lstm.flatten_parameters()
        hiddens, output = self.lstm(packed_embedded)
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True)
        #hiddens -> (batch, seq_len, n_directions* hidden_size)
        
        out1, out2 = torch.chunk(hiddens, 2, dim=2)
        final_out=[]
        for idx in range(out1.shape[0]):
            final_out.append(torch.cat((out1[idx, sent_lens[idx]-1, :], out2[idx, 0, :])))
        final_out = torch.stack(final_out,0)
        return final_out


class BiLSTM_xpath(nn.Module):

    def __init__(self, vocab_size, emb_dim, hidden_size):
        super(BiLSTM_xpath, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=True, bidirectional=True)

    def forward(self, inputs, seq_lens):
        #inputs = (batch,seq_len)
        xpath_embedding = self.embedding(inputs)
        # xpath_embedding = (batch,seq_len, embedding_dim)
        
        packed_embedded = pack_padded_sequence(xpath_embedding, seq_lens, batch_first=True, enforce_sorted=False)
        # self.lstm.flatten_parameters()
        hiddens, output = self.lstm(packed_embedded)
        #hiddens -> (batch, seq_len, n_directions* hidden_size)
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True)

        out1, out2 = torch.chunk(hiddens, 2, dim=2)
        final_out=[]
        for idx in range(out1.shape[0]):
            final_out.append(torch.cat((out1[idx, seq_lens[idx]-1, :], out2[idx, 0, :])))
        final_out = torch.stack(final_out,0)
        return final_out