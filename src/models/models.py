import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import collections


from src.utils.constants import PAD
from src.data.pytorch_datasets import pitch_to_ix, ks_to_ix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Device: " + str(device))

#https://github.com/sooftware/attentions/blob/master/attentions.py
class AdditiveAttention(nn.Module):
    """
     Applies a additive attention (bahdanau) mechanism on the output features from the decoder.
     Additive attention proposed in "Neural Machine Translation by Jointly Learning to Align and Translate" paper.
     Args:
         hidden_dim (int): dimesion of hidden state vector
     Inputs: query, value
         - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
         - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.
     Returns: context, attn
         - **context**: tensor containing the context vector from attention mechanism.
         - **attn**: tensor containing the alignment from the encoder outputs.
     Reference:
         - **Neural Machine Translation by Jointly Learning to Align and Translate**: https://arxiv.org/abs/1409.0473
    """
    def __init__(self, hidden_dim: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query: Tensor, key: Tensor, value: Tensor):# -> Tuple[Tensor, Tensor]:
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)
        context = torch.sum(context, dim=1,  keepdim=True)
        return context, attn

class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """
    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()
        self.normalize = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=True) #False?
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)


    def forward(self, query: Tensor, value: Tensor): #-> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(self.query_proj(query), self.value_proj(value).transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.sum(torch.bmm(attn, value), dim=1,  keepdim=True)

        return context, attn

class DotProductAttention_nosum(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """
    def __init__(self, hidden_dim):
        super(DotProductAttention2, self).__init__()
        self.normalize = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=True) #False?
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, query: Tensor, value: Tensor): #-> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(self.query_proj(query), self.value_proj(value).transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, self.out_proj(value))
        #context = torch.sum(torch.bmm(attn, value), dim=1,  keepdim=True)

        return context, attn

class PKSpellHierarchical_app1(nn.Module):
    """Models that adds Hierarchical attention"""

    def __init__(
        self,
        input_dim=17,
        hidden_dim=300,
        pitch_to_ix=pitch_to_ix,
        ks_to_ix=ks_to_ix,
        hidden_dim2=24,
        rnn_depth=1,
        dropout=None,
        dropout2=None,
        cell_type="GRU",
        bidirectional=True,
        mode="both",
    ):
        super(PKSpellHierarchical_app1, self).__init__()

        self.n_out_pitch = len(pitch_to_ix)
        self.n_out_ks = len(ks_to_ix)

        if hidden_dim % 2 != 0:
            raise ValueError("Hidden_dim must be an even integer")
        if hidden_dim2 % 2 != 0:
            raise ValueError("Hidden_dim2 must be an even integer")
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2

        if cell_type == "GRU":
            rnn_cell = nn.GRU
        elif cell_type == "LSTM":
            rnn_cell = nn.LSTM
        else:
            raise ValueError(f"Unknown RNN cell type: {cell_type}")

        # RNN layer.
        self.rnn = rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            bidirectional=bidirectional,
            num_layers=rnn_depth,
        )


        self.hier_hidden = 256
        self.hier_rnn = rnn_cell(
            input_size=hidden_dim,
            hidden_size=self.hier_hidden//2,
            bidirectional=True,
            num_layers=1,
        )

        self.att_layer = DotProductAttention(self.hier_hidden)

        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if dropout2 is not None and dropout2 > 0:
            self.dropout2 = nn.Dropout(p=dropout2)
        else:
            self.dropout2 = None

        # Output layers.
        self.top_layer_pitch = nn.Linear(hidden_dim+self.hier_hidden, self.n_out_pitch)
        self.top_layer_ks = nn.Linear(self.hier_hidden, self.n_out_ks)


        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])
        self.mode = mode

    def compute_outputs(self, sentences, sentences_len, eoM):


        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        rnn_out, _ = self.rnn(sentences)

        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        if self.dropout is not None:
            rnn_out = self.dropout(rnn_out)

        context_list = []

        for i, s, eom, l in zip(range(len(sentences_len)), torch.transpose(rnn_out,0,1),torch.transpose(eoM,0,1),sentences_len):
            nz = torch.nonzero(eom).squeeze()
            lengths = torch.diff(nz.to(device),prepend=torch.tensor([-1]).to(device))
            
            sentences_split = torch.tensor_split(s[:l.int()], nz.cpu())
            
            sentences_split_pad = nn.utils.rnn.pad_sequence(sentences_split,batch_first=False)

            packed = nn.utils.rnn.pack_padded_sequence(sentences_split_pad, lengths.cpu(),enforce_sorted=False)

            rnn_o, h_n = self.hier_rnn(packed)

            rnn_o, _ = nn.utils.rnn.pad_packed_sequence(rnn_o)

            attn_output, attn_output_weights = self.att_layer(torch.transpose(rnn_o,0,1),torch.transpose(rnn_o,0,1))

            context = attn_output.squeeze()

            context = torch.repeat_interleave(context, lengths.int(), dim=0)
            context_list.append(context)



        out_context = nn.utils.rnn.pad_sequence(context_list,batch_first=True)

        stacked = torch.cat((rnn_out,torch.transpose(out_context,0,1)),dim=2)

        out_pitch = self.top_layer_pitch(stacked)

        out_ks = self.top_layer_ks(torch.transpose(out_context,0,1))

        return out_pitch, out_ks

    def forward(self, sentences, pitches, keysignatures, sentences_len, eoM):
        # First computes the predictions, and then the loss function.

        # Compute the outputs. The shape is (max_len, n_sentences, n_labels).
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len, eoM)

        # Flatten the outputs and the gold-standard labels, to compute the loss.
        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
        scores_pitch = scores_pitch.view(-1, self.n_out_pitch)
        scores_ks = scores_ks.view(-1, self.n_out_ks)
        pitches = pitches.view(-1)
        keysignatures = keysignatures.view(-1)

        if self.mode == "both":
            loss = self.loss_pitch(scores_pitch, pitches) + self.loss_ks(
                scores_ks, keysignatures
            )
        elif self.mode == "ks":
            loss = self.loss_ks(scores_ks, keysignatures)
        elif self.mode == "ps":
            loss = self.loss_pitch(scores_pitch, pitches)
        return loss

    def predict(self, sentences, sentences_len, eoM):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len, eoM)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted_pitch = scores_pitch.argmax(dim=2)
        predicted_ks = scores_ks.argmax(dim=2)
        return (
            [
                predicted_pitch[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
            [
                predicted_ks[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
        )

class PKSpellHierarchical_app2(nn.Module):
    """Models that adds Hierarchical attention"""

    def __init__(
        self,
        input_dim=17,
        hidden_dim=300,
        pitch_to_ix=pitch_to_ix,
        ks_to_ix=ks_to_ix,
        hidden_dim2=24,
        rnn_depth=1,
        dropout=None,
        dropout2=None,
        cell_type="GRU",
        bidirectional=True,
        mode="both",
    ):
        super(PKSpellHierarchical_app2, self).__init__()

        self.n_out_pitch = len(pitch_to_ix)
        self.n_out_ks = len(ks_to_ix)

        if hidden_dim % 2 != 0:
            raise ValueError("Hidden_dim must be an even integer")
        if hidden_dim2 % 2 != 0:
            raise ValueError("Hidden_dim2 must be an even integer")
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2

        if cell_type == "GRU":
            rnn_cell = nn.GRU
        elif cell_type == "LSTM":
            rnn_cell = nn.LSTM
        else:
            raise ValueError(f"Unknown RNN cell type: {cell_type}")

        # RNN layer.
        self.rnn = rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            bidirectional=bidirectional,
            num_layers=rnn_depth,
        )


        self.hier_hidden = 256
        self.hier_rnn = rnn_cell(
            input_size=input_dim,#input_dim hidden_dim
            hidden_size=self.hier_hidden//2,
            bidirectional=True,
            num_layers=1,
        )


        self.att_layer = DotProductAttention(self.hier_hidden)

        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if dropout2 is not None and dropout2 > 0:
            self.dropout2 = nn.Dropout(p=dropout2)
        else:
            self.dropout2 = None

        # Output layers.
        self.top_layer_pitch = nn.Linear(hidden_dim+self.hier_hidden, self.n_out_pitch)
        self.top_layer_ks = nn.Linear(self.hier_hidden, self.n_out_ks)


        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])
        self.mode = mode

    def compute_outputs(self, sentences, sentences_len, eoM):

        context_list = []

        for i, s, eom, l in zip(range(len(sentences_len)), torch.transpose(sentences,0,1),torch.transpose(eoM,0,1),sentences_len):
            nz = torch.nonzero(eom).squeeze()
            lengths = torch.diff(nz.to(device),prepend=torch.tensor([-1]).to(device))
            
            sentences_split = torch.tensor_split(s[:l.int()], nz.cpu())
            
            sentences_split_pad = nn.utils.rnn.pad_sequence(sentences_split,batch_first=False)

            packed = nn.utils.rnn.pack_padded_sequence(sentences_split_pad, lengths.cpu(),enforce_sorted=False)

            rnn_o, h_n = self.hier_rnn(packed)

            rnn_o, _ = nn.utils.rnn.pad_packed_sequence(rnn_o)

            attn_output, attn_output_weights = self.att_layer(torch.transpose(rnn_o,0,1),torch.transpose(rnn_o,0,1))

            context = attn_output.squeeze()

            context = torch.repeat_interleave(context, lengths.int(), dim=0)
            context_list.append(context)


        out_context = nn.utils.rnn.pad_sequence(context_list,batch_first=True)
        
        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        rnn_out, _ = self.rnn(sentences)

        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        if self.dropout is not None:
            rnn_out = self.dropout(rnn_out)

        stacked = torch.cat((rnn_out,torch.transpose(out_context,0,1)),dim=2)

        out_pitch = self.top_layer_pitch(stacked)

        out_ks = self.top_layer_ks(torch.transpose(out_context,0,1))

        return out_pitch, out_ks

    def forward(self, sentences, pitches, keysignatures, sentences_len, eoM):
        # First computes the predictions, and then the loss function.

        # Compute the outputs. The shape is (max_len, n_sentences, n_labels).
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len, eoM)

        # Flatten the outputs and the gold-standard labels, to compute the loss.
        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
        scores_pitch = scores_pitch.view(-1, self.n_out_pitch)
        scores_ks = scores_ks.view(-1, self.n_out_ks)
        pitches = pitches.view(-1)
        keysignatures = keysignatures.view(-1)

        if self.mode == "both":
            loss = self.loss_pitch(scores_pitch, pitches) + self.loss_ks(
                scores_ks, keysignatures
            )
        elif self.mode == "ks":
            loss = self.loss_ks(scores_ks, keysignatures)
        elif self.mode == "ps":
            loss = self.loss_pitch(scores_pitch, pitches)
        return loss

    def predict(self, sentences, sentences_len, eoM):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len, eoM)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted_pitch = scores_pitch.argmax(dim=2)
        predicted_ks = scores_ks.argmax(dim=2)
        return (
            [
                predicted_pitch[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
            [
                predicted_ks[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
        )

class PKSpellHierarchical_app3(nn.Module):
    """Models that adds Hierarchical attention"""

    def __init__(
        self,
        input_dim=17,
        hidden_dim=300,
        pitch_to_ix=pitch_to_ix,
        ks_to_ix=ks_to_ix,
        hidden_dim2=24,
        rnn_depth=1,
        dropout=None,
        dropout2=None,
        cell_type="GRU",
        bidirectional=True,
        mode="both",
    ):
        super(PKSpellHierarchical_app3, self).__init__()

        self.n_out_pitch = len(pitch_to_ix)
        self.n_out_ks = len(ks_to_ix)

        if hidden_dim % 2 != 0:
            raise ValueError("Hidden_dim must be an even integer")
        if hidden_dim2 % 2 != 0:
            raise ValueError("Hidden_dim2 must be an even integer")
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2

        if cell_type == "GRU":
            rnn_cell = nn.GRU
        elif cell_type == "LSTM":
            rnn_cell = nn.LSTM
        else:
            raise ValueError(f"Unknown RNN cell type: {cell_type}")

        # RNN layer.
        self.rnn = rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            bidirectional=bidirectional,
            num_layers=rnn_depth,
        )

        self.hier_hidden = 256
        self.hier_rnn = rnn_cell(
            input_size=hidden_dim,
            hidden_size=self.hier_hidden,#//2,
            bidirectional=False,
            num_layers=1,
        )

        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if dropout2 is not None and dropout2 > 0:
            self.dropout2 = nn.Dropout(p=dropout2)
        else:
            self.dropout2 = None

        # Output layers.
        self.top_layer_pitch = nn.Linear(hidden_dim+self.hier_hidden, self.n_out_pitch)
        self.top_layer_ks = nn.Linear(self.hier_hidden, self.n_out_ks)

        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])
        self.mode = mode

    def compute_outputs(self, sentences, sentences_len, eoM):


        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        rnn_out, _ = self.rnn(sentences)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)


        if self.dropout is not None:
            rnn_out = self.dropout(rnn_out)


        context_list = []
        for i, s, eom, l in zip(range(len(sentences_len)), torch.transpose(rnn_out,0,1),torch.transpose(eoM,0,1),sentences_len):
            nz = torch.nonzero(eom).squeeze()
            sentences_split = torch.tensor_split(s[:l.int()], nz.cpu())
            lengths = torch.diff(nz.to(device),prepend=torch.tensor([-1]).to(device))

            sentences_split_pad = nn.utils.rnn.pad_sequence(sentences_split,batch_first=False)
            packed = nn.utils.rnn.pack_padded_sequence(sentences_split_pad, lengths.cpu(),enforce_sorted=False)
            rnn_o, h_n = self.hier_rnn(packed)

            context = h_n.squeeze()

            context = torch.repeat_interleave(context, lengths.int(), dim=0)

            context_list.append(context)


        out_context = nn.utils.rnn.pad_sequence(context_list,batch_first=True)

        stacked = torch.cat((rnn_out,torch.transpose(out_context,0,1)),dim=2)
        out_pitch = self.top_layer_pitch(stacked)

        out_ks = self.top_layer_ks(torch.transpose(out_context,0,1))

        return out_pitch, out_ks

    def forward(self, sentences, pitches, keysignatures, sentences_len, eoM):
        # First computes the predictions, and then the loss function.

        # Compute the outputs. The shape is (max_len, n_sentences, n_labels).
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len, eoM)

        # Flatten the outputs and the gold-standard labels, to compute the loss.
        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
        scores_pitch = scores_pitch.view(-1, self.n_out_pitch)
        scores_ks = scores_ks.view(-1, self.n_out_ks)
        pitches = pitches.view(-1)
        keysignatures = keysignatures.view(-1)

        if self.mode == "both":
            loss = self.loss_pitch(scores_pitch, pitches) + self.loss_ks(
                scores_ks, keysignatures
            )
        elif self.mode == "ks":
            loss = self.loss_ks(scores_ks, keysignatures)
        elif self.mode == "ps":
            loss = self.loss_pitch(scores_pitch, pitches)
        return loss

    def predict(self, sentences, sentences_len, eoM):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len, eoM)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted_pitch = scores_pitch.argmax(dim=2)
        predicted_ks = scores_ks.argmax(dim=2)
        return (
            [
                predicted_pitch[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
            [
                predicted_ks[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
        )

class PKSpellHierarchical_app4(nn.Module):
    """Models that adds Hierarchical attention"""

    def __init__(
        self,
        input_dim=17,
        hidden_dim=300,
        pitch_to_ix=pitch_to_ix,
        ks_to_ix=ks_to_ix,
        hidden_dim2=24,
        rnn_depth=1,
        dropout=None,
        dropout2=None,
        cell_type="GRU",
        bidirectional=True,
        mode="both",
    ):
        super(PKSpellHierarchical_app4, self).__init__()

        self.n_out_pitch = len(pitch_to_ix)
        self.n_out_ks = len(ks_to_ix)

        if hidden_dim % 2 != 0:
            raise ValueError("Hidden_dim must be an even integer")
        if hidden_dim2 % 2 != 0:
            raise ValueError("Hidden_dim2 must be an even integer")
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2

        if cell_type == "GRU":
            rnn_cell = nn.GRU
        elif cell_type == "LSTM":
            rnn_cell = nn.LSTM
        else:
            raise ValueError(f"Unknown RNN cell type: {cell_type}")

        # RNN layer.
        self.rnn = rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            bidirectional=bidirectional,
            num_layers=rnn_depth,
        )

        self.hier_hidden = 256
        self.hier_rnn = rnn_cell(
            input_size=hidden_dim,#input_dim hidden_dim
            hidden_size=self.hier_hidden,#//2,
            bidirectional=False,
            num_layers=1,
        )

        self.att_layer = nn.MultiheadAttention(self.hier_hidden, num_heads=4, batch_first=True)

        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if dropout2 is not None and dropout2 > 0:
            self.dropout2 = nn.Dropout(p=dropout2)
        else:
            self.dropout2 = None

        # Output layers.
        self.top_layer_pitch = nn.Linear(hidden_dim+self.hier_hidden, self.n_out_pitch)
        self.top_layer_ks = nn.Linear(self.hier_hidden, self.n_out_ks)

        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])
        self.mode = mode

    def compute_outputs(self, sentences, sentences_len, eoM):


        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        rnn_out, _ = self.rnn(sentences)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        if self.dropout is not None:
            rnn_out = self.dropout(rnn_out)


        context_list = []
        for i, s, eom, l in zip(range(len(sentences_len)), torch.transpose(rnn_out,0,1),torch.transpose(eoM,0,1),sentences_len):
            nz = torch.nonzero(eom).squeeze()
            sentences_split = torch.tensor_split(s[:l.int()], nz.cpu())
            lengths = torch.diff(nz.to(device),prepend=torch.tensor([-1]).to(device))

            sentences_split_pad = nn.utils.rnn.pad_sequence(sentences_split,batch_first=False)
            packed = nn.utils.rnn.pack_padded_sequence(sentences_split_pad, lengths.cpu(),enforce_sorted=False)
            rnn_o, h_n = self.hier_rnn(packed)

            attn_output, attn_output_weights = self.att_layer(h_n,h_n,h_n)

            context = attn_output.squeeze()

            context = torch.repeat_interleave(context, lengths.int(), dim=0)

            context_list.append(context)


        out_context = nn.utils.rnn.pad_sequence(context_list,batch_first=True)

        stacked = torch.cat((rnn_out,torch.transpose(out_context,0,1)),dim=2)
        out_pitch = self.top_layer_pitch(stacked)

        out_ks = self.top_layer_ks(torch.transpose(out_context,0,1))

        return out_pitch, out_ks

    def forward(self, sentences, pitches, keysignatures, sentences_len, eoM):
        # First computes the predictions, and then the loss function.

        # Compute the outputs. The shape is (max_len, n_sentences, n_labels).
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len, eoM)

        # Flatten the outputs and the gold-standard labels, to compute the loss.
        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
        scores_pitch = scores_pitch.view(-1, self.n_out_pitch)
        scores_ks = scores_ks.view(-1, self.n_out_ks)
        pitches = pitches.view(-1)
        keysignatures = keysignatures.view(-1)

        if self.mode == "both":
            loss = self.loss_pitch(scores_pitch, pitches) + self.loss_ks(
                scores_ks, keysignatures
            )
        elif self.mode == "ks":
            loss = self.loss_ks(scores_ks, keysignatures)
        elif self.mode == "ps":
            loss = self.loss_pitch(scores_pitch, pitches)
        return loss

    def predict(self, sentences, sentences_len, eoM):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len, eoM)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted_pitch = scores_pitch.argmax(dim=2)
        predicted_ks = scores_ks.argmax(dim=2)
        return (
            [
                predicted_pitch[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
            [
                predicted_ks[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
        )

class PKSpellHierarchical_app5(nn.Module):
    """Models that only has one RNN and uses last element of every measure for KS prediction"""

    def __init__(
        self,
        input_dim=17,
        hidden_dim=300,
        pitch_to_ix=pitch_to_ix,
        ks_to_ix=ks_to_ix,
        hidden_dim2=24,
        rnn_depth=1,
        dropout=None,
        dropout2=None,
        cell_type="GRU",
        bidirectional=True,
        mode="both",
    ):
        super(PKSpellHierarchical_app5, self).__init__()

        self.n_out_pitch = len(pitch_to_ix)
        self.n_out_ks = len(ks_to_ix)

        if hidden_dim % 2 != 0:
            raise ValueError("Hidden_dim must be an even integer")
        if hidden_dim2 % 2 != 0:
            raise ValueError("Hidden_dim2 must be an even integer")
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2

        if cell_type == "GRU":
            rnn_cell = nn.GRU
        elif cell_type == "LSTM":
            rnn_cell = nn.LSTM
        else:
            raise ValueError(f"Unknown RNN cell type: {cell_type}")

        # RNN layer.
        self.rnn = rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            bidirectional=bidirectional,
            num_layers=rnn_depth,
        )


        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if dropout2 is not None and dropout2 > 0:
            self.dropout2 = nn.Dropout(p=dropout2)
        else:
            self.dropout2 = None

        # Output layers.

        self.top_layer_pitch = nn.Linear(hidden_dim, self.n_out_pitch)
        self.top_layer_ks = nn.Linear(hidden_dim, self.n_out_ks)


        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])
        self.mode = mode

    def compute_outputs(self, sentences, sentences_len, eoM):

        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        rnn_out, _ = self.rnn(sentences)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        if self.dropout is not None:
            rnn_out = self.dropout(rnn_out)

        context_list = []

        for i, s, eom, l in zip(range(len(sentences_len)), torch.transpose(rnn_out,0,1),torch.transpose(eoM,0,1),sentences_len):
            
            nz = torch.nonzero(eom).squeeze()
            lengths = torch.diff(nz.to(device),prepend=torch.tensor([-1]).to(device))
            context = torch.repeat_interleave(s[nz], lengths.int(), dim=0)
            context_list.append(context)

        out_context = nn.utils.rnn.pad_sequence(context_list,batch_first=True)

        out_pitch = self.top_layer_pitch(rnn_out)

        out_ks = self.top_layer_ks(torch.transpose(out_context,0,1))

        return out_pitch, out_ks

    def forward(self, sentences, pitches, keysignatures, sentences_len, eoM):
        # First computes the predictions, and then the loss function.

        # Compute the outputs. The shape is (max_len, n_sentences, n_labels).
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len, eoM)

        # Flatten the outputs and the gold-standard labels, to compute the loss.
        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
        scores_pitch = scores_pitch.view(-1, self.n_out_pitch)
        scores_ks = scores_ks.view(-1, self.n_out_ks)
        pitches = pitches.view(-1)
        keysignatures = keysignatures.view(-1)

        if self.mode == "both":
            loss = self.loss_pitch(scores_pitch, pitches) + self.loss_ks(
                scores_ks, keysignatures
            )
        elif self.mode == "ks":
            loss = self.loss_ks(scores_ks, keysignatures)
        elif self.mode == "ps":
            loss = self.loss_pitch(scores_pitch, pitches)
        return loss

    def predict(self, sentences, sentences_len, eoM):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len, eoM)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted_pitch = scores_pitch.argmax(dim=2)
        predicted_ks = scores_ks.argmax(dim=2)
        return (
            [
                predicted_pitch[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
            [
                predicted_ks[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
        )

class PKSpellHierarchical_app6(nn.Module):
    """Models that adds Hierarchical attention"""

    def __init__(
        self,
        input_dim=17,
        hidden_dim=300,
        pitch_to_ix=pitch_to_ix,
        ks_to_ix=ks_to_ix,
        hidden_dim2=24,
        rnn_depth=1,
        dropout=None,
        dropout2=None,
        cell_type="GRU",
        bidirectional=True,
        mode="both",
    ):
        super(PKSpellHierarchical_app6, self).__init__()

        self.n_out_pitch = len(pitch_to_ix)
        self.n_out_ks = len(ks_to_ix)

        if hidden_dim % 2 != 0:
            raise ValueError("Hidden_dim must be an even integer")
        if hidden_dim2 % 2 != 0:
            raise ValueError("Hidden_dim2 must be an even integer")
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2

        if cell_type == "GRU":
            rnn_cell = nn.GRU
        elif cell_type == "LSTM":
            rnn_cell = nn.LSTM
        else:
            raise ValueError(f"Unknown RNN cell type: {cell_type}")

        # RNN layer.
        self.rnn = rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            bidirectional=bidirectional,
            num_layers=rnn_depth,
        )


        self.att_layer1 = DotProductAttention(hidden_dim)
        self.att_layer2 = nn.MultiheadAttention(hidden_dim, num_heads=2, batch_first=False)

        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if dropout2 is not None and dropout2 > 0:
            self.dropout2 = nn.Dropout(p=dropout2)
        else:
            self.dropout2 = None

        # Output layers.
        self.top_layer_pitch = nn.Linear(hidden_dim+hidden_dim, self.n_out_pitch)
        self.top_layer_ks = nn.Linear(hidden_dim, self.n_out_ks)

        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])
        self.mode = mode

    def compute_outputs(self, sentences, sentences_len, eoM):

        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        rnn_out, _ = self.rnn(sentences)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        if self.dropout is not None:
            rnn_out = self.dropout(rnn_out)

        context_list = []

        for i, s, eom, l in zip(range(len(sentences_len)), torch.transpose(rnn_out,0,1),torch.transpose(eoM,0,1),sentences_len):
            nz = torch.nonzero(eom).squeeze()

            sentences_split = torch.tensor_split(s[:l.int()], nz.cpu())

            lengths = torch.diff(nz.to(device),prepend=torch.tensor([-1]).to(device))

            sentences_split_pad = nn.utils.rnn.pad_sequence(sentences_split,batch_first=False)[:,:len(nz),:]
            
            sentences_split_pad = torch.transpose(sentences_split_pad,0,1)

            context, _ = self.att_layer1(sentences_split_pad,sentences_split_pad)
            
            context = torch.transpose(context,0,1)

            context, _ = self.att_layer2(context,context,context)

            context = context.squeeze()
            context = torch.repeat_interleave(context, lengths.int(), dim=0)

            context_list.append(context)

        out_context = nn.utils.rnn.pad_sequence(context_list,batch_first=True)

        stacked = torch.cat((rnn_out,torch.transpose(out_context,0,1)),dim=2)
        out_pitch = self.top_layer_pitch(stacked)

        out_ks = self.top_layer_ks(torch.transpose(out_context,0,1))

        return out_pitch, out_ks

    def forward(self, sentences, pitches, keysignatures, sentences_len, eoM):
        # First computes the predictions, and then the loss function.

        # Compute the outputs. The shape is (max_len, n_sentences, n_labels).
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len, eoM)

        # Flatten the outputs and the gold-standard labels, to compute the loss.
        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
        scores_pitch = scores_pitch.view(-1, self.n_out_pitch)
        scores_ks = scores_ks.view(-1, self.n_out_ks)
        pitches = pitches.view(-1)
        keysignatures = keysignatures.view(-1)

        if self.mode == "both":
            loss = self.loss_pitch(scores_pitch, pitches) + 2*self.loss_ks(
                scores_ks, keysignatures
            )
        elif self.mode == "ks":
            loss = self.loss_ks(scores_ks, keysignatures)
        elif self.mode == "ps":
            loss = self.loss_pitch(scores_pitch, pitches)
        return loss

    def predict(self, sentences, sentences_len, eoM):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len, eoM)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted_pitch = scores_pitch.argmax(dim=2)
        predicted_ks = scores_ks.argmax(dim=2)
        return (
            [
                predicted_pitch[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
            [
                predicted_ks[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
        )

class PKSpellHierarchical_app7(nn.Module):
    """Models that adds Hierarchical attention"""

    def __init__(
        self,
        input_dim=17,
        hidden_dim=300,
        pitch_to_ix=pitch_to_ix,
        ks_to_ix=ks_to_ix,
        hidden_dim2=24,
        rnn_depth=1,
        dropout=None,
        dropout2=None,
        cell_type="GRU",
        bidirectional=True,
        mode="both",
    ):
        super(PKSpellHierarchical_app7, self).__init__()

        self.n_out_pitch = len(pitch_to_ix)
        self.n_out_ks = len(ks_to_ix)

        if hidden_dim % 2 != 0:
            raise ValueError("Hidden_dim must be an even integer")
        if hidden_dim2 % 2 != 0:
            raise ValueError("Hidden_dim2 must be an even integer")
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2

        if cell_type == "GRU":
            rnn_cell = nn.GRU
        elif cell_type == "LSTM":
            rnn_cell = nn.LSTM
        else:
            raise ValueError(f"Unknown RNN cell type: {cell_type}")

        # RNN layer.
        self.rnn = rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            bidirectional=bidirectional,
            num_layers=rnn_depth,
        )

        self.att_layer1 = DotProductAttention(hidden_dim)

        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if dropout2 is not None and dropout2 > 0:
            self.dropout2 = nn.Dropout(p=dropout2)
        else:
            self.dropout2 = None

        # Output layers.

        self.top_layer_pitch = nn.Linear(hidden_dim+hidden_dim, self.n_out_pitch)
        self.top_layer_ks = nn.Linear(hidden_dim, self.n_out_ks)


        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])
        self.mode = mode

    def compute_outputs(self, sentences, sentences_len, eoM):

        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        rnn_out, _ = self.rnn(sentences)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        if self.dropout is not None:
            rnn_out = self.dropout(rnn_out)

        context_list = []

        for i, s, eom, l in zip(range(len(sentences_len)), torch.transpose(rnn_out,0,1),torch.transpose(eoM,0,1),sentences_len):
            nz = torch.nonzero(eom).squeeze()

            sentences_split = torch.tensor_split(s[:l.int()], nz.cpu())

            lengths = torch.diff(nz.to(device),prepend=torch.tensor([-1]).to(device))

            sentences_split_pad = nn.utils.rnn.pad_sequence(sentences_split,batch_first=False)[:,:len(nz),:]

            context, _ = self.att_layer1(torch.transpose(sentences_split_pad,0,1),torch.transpose(sentences_split_pad,0,1))

            context = context.squeeze()
            context = torch.repeat_interleave(context, lengths.int(), dim=0)

            context_list.append(context)

        out_context = nn.utils.rnn.pad_sequence(context_list,batch_first=True)

        stacked = torch.cat((rnn_out,torch.transpose(out_context,0,1)),dim=2)
        out_pitch = self.top_layer_pitch(stacked)

        out_ks = self.top_layer_ks(torch.transpose(out_context,0,1))

        return out_pitch, out_ks

    def forward(self, sentences, pitches, keysignatures, sentences_len, eoM):
        # First computes the predictions, and then the loss function.

        # Compute the outputs. The shape is (max_len, n_sentences, n_labels).
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len, eoM)

        # Flatten the outputs and the gold-standard labels, to compute the loss.
        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
        scores_pitch = scores_pitch.view(-1, self.n_out_pitch)
        scores_ks = scores_ks.view(-1, self.n_out_ks)
        pitches = pitches.view(-1)
        keysignatures = keysignatures.view(-1)

        if self.mode == "both":
            loss = self.loss_pitch(scores_pitch, pitches) + 2*self.loss_ks(
                scores_ks, keysignatures
            )
        elif self.mode == "ks":
            loss = self.loss_ks(scores_ks, keysignatures)
        elif self.mode == "ps":
            loss = self.loss_pitch(scores_pitch, pitches)
        return loss

    def predict(self, sentences, sentences_len, eoM):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len, eoM)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted_pitch = scores_pitch.argmax(dim=2)
        predicted_ks = scores_ks.argmax(dim=2)
        return (
            [
                predicted_pitch[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
            [
                predicted_ks[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
        )
