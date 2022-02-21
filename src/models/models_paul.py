import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import collections

from src.utils.constants import PAD
from src.data.pytorch_datasets import pitch_to_ix, ks_to_ix


#https://github.com/sooftware/attentions/blob/master/attentions.py
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

class PKSpellHierarchical(nn.Module):
    """Models that decouples key signature estimation from pitch spelling by adding a second RNN.

    This model reached state of the art performances for pitch spelling.
    """

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
        super(PKSpellHierarchical, self).__init__()

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
        self.rnn2 = rnn_cell(
            input_size=hidden_dim,
            hidden_size=hidden_dim2 // 2 if bidirectional else hidden_dim2,
            bidirectional=bidirectional,
            num_layers=rnn_depth,
        )
        self.hier_hidden = 32
        self.hier_rnn = rnn_cell(
            input_size=input_dim,
            hidden_size=self.hier_hidden//2,
            bidirectional=True,
            num_layers=1,
        )

        self.hier_rnn2 = rnn_cell(
            input_size=self.hier_hidden,
            hidden_size=self.hier_hidden//2,
            bidirectional=True,
            num_layers=1,
            batch_first=True
        )

        self.att_layer1 = DotProductAttention(self.hier_hidden)
        self.att_layer2 = DotProductAttention(self.hier_hidden)

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
        self.top_layer_ks = nn.Linear(hidden_dim2+self.hier_hidden, self.n_out_ks)

        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])
        self.mode = mode

    def compute_outputs(self, sentences, sentences_len, eoM):

        #print("s",sentences.shape)
        #print("sl",sentences_len.shape)
        #print("eoM",eoM.shape,torch.transpose(eoM,0,1).shape)
        #print(sentences_len)#
        #print(eoM)
        #nz = torch.nonzero(torch.transpose(eoM,0,1), as_tuple=True)
        #print("nz",np.array(nz).shape,torch.transpose(eoM,0,1).nonzero())
        #print(nz)

        #iterate through batch to get hierarchical representation

        hier_out = torch.empty((len(sentences_len), self.hier_hidden), device='cuda')
        for i, s, eom, l in zip(range(len(sentences_len)), torch.transpose(sentences,0,1),torch.transpose(eoM,0,1),sentences_len):
            nz = torch.nonzero(eom).squeeze()
            #print("i",i)
            #print("nz",nz.shape)
            #print(len(nz),nz)
            #print("s",s.shape)
            #print("l",l)
            sentences_split = torch.tensor_split(s[:l.int()], nz.clone().cpu())
            #print(len(sentences_split))
            if nz[0]==0:
                lengths = torch.diff(nz.cuda(),append=torch.tensor([l]).cuda())
            else:
                #print(nz.device,torch.tensor([0]).cuda().device,torch.tensor([l]).cuda().device)
                lengths = torch.diff(nz.cuda(),prepend=torch.tensor([0]).cuda(),append=torch.tensor([l]).cuda())
            #print(len(lengths),lengths)
            #print(np.array(sentences_split).shape)
            sentences_split_pad = nn.utils.rnn.pad_sequence(sentences_split,batch_first=False)
            packed = nn.utils.rnn.pack_padded_sequence(sentences_split_pad, lengths.cpu(),enforce_sorted=False)

            rnn_o, _ = self.hier_rnn(packed)

            rnn_o, _ = nn.utils.rnn.pad_packed_sequence(rnn_o)


            #print("rnn_o",rnn_o.shape)
            context, _ = self.att_layer1(rnn_o, rnn_o)
            #print("context",context.shape)

            rnn_o, _ = self.hier_rnn2(context)
            rnn_o = torch.transpose(rnn_o,0,1)
            #print("rnn_o",rnn_o.shape)

            context, _ = self.att_layer2(rnn_o, rnn_o)
            #print("context",context.shape)

            hier_out[i] = context.squeeze()
            #break
        #print(hier_out)
        hier_out = hier_out.unsqueeze(0).expand(sentences.size(0),-1,-1)


        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        #print("se",sentences.shape)
        rnn_out, _ = self.rnn(sentences)
        #print("ro",rnn_out.shape)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)
        #print("ro",rnn_out.shape)


        if self.dropout is not None:
            rnn_out = self.dropout(rnn_out)
        #print("rooO",rnn_out.shape)
        #print("dss")
        #print(rnn_out.device, hier_out.device)
        stacked = torch.cat((rnn_out,hier_out),dim=2)
        #print(stacked.shape)
        out_pitch = self.top_layer_pitch(stacked)
        #out_pitch = self.top_layer_pitch(rnn_out)


        # pass the ks information into the second rnn
        rnn_out = nn.utils.rnn.pack_padded_sequence(rnn_out, sentences_len)
        rnn_out, _ = self.rnn2(rnn_out)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        if self.dropout2 is not None:
            rnn_out = self.dropout2(rnn_out)

        #print("sda",rnn_out.shape)
        stacked = torch.cat((rnn_out,hier_out),dim=2)
        out_ks = self.top_layer_ks(stacked)
        #out_ks = self.top_layer_ks(rnn_out)

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






# transformer code from https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51

#def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
#    temp = query.bmm(key.transpose(1, 2))
#    scale = query.size(-1) ** 0.5
#    softmax = f.softmax(temp / scale, dim=-1)
#    return softmax.bmm(value)
#
#
#class AttentionHead(nn.Module):
#    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
#        super().__init__()
#        self.q = nn.Linear(dim_in, dim_q)
#        self.k = nn.Linear(dim_in, dim_k)
#        self.v = nn.Linear(dim_in, dim_k)
#
#    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
#        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))
#
#
#class MultiHeadAttention(nn.Module):
#    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
#        super().__init__()
#        self.heads = nn.ModuleList(
#            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
#        )
#        self.linear = nn.Linear(num_heads * dim_k, dim_in)
#
#    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
#        return self.linear(
#            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
#        )
#
#
#def position_encoding(
#    seq_len: int, dim_model: int, device: torch.device = torch.device("cpu"),
#) -> Tensor:
#    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
#    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
#    phase = pos / (1e4 ** (dim // dim_model))
#
#    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
#
#
#def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
#    return nn.Sequential(
#        nn.Linear(dim_input, dim_feedforward),
#        nn.ReLU(),
#        nn.Linear(dim_feedforward, dim_input),
#    )
#
#class Residual(nn.Module):
#    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
#        super().__init__()
#        self.sublayer = sublayer
#        self.norm = nn.LayerNorm(dimension)
#        self.dropout = nn.Dropout(dropout)
#
#    def forward(self, *tensors: Tensor) -> Tensor:
#        # Assume that the "query" tensor is given first, so we can compute the
#        # residual.  This matches the signature of 'MultiHeadAttention'.
#        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))
#
#class TransformerEncoderLayer(nn.Module):
#    def __init__(
#        self,
#        dim_model: int = 512,
#        num_heads: int = 6,
#        dim_feedforward: int = 2048,
#        dropout: float = 0.1,
#    ):
#        super().__init__()
#        dim_q = dim_k = max(dim_model // num_heads, 1)
#        self.attention = Residual(
#            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
#            dimension=dim_model,
#            dropout=dropout,
#        )
#        self.feed_forward = Residual(
#            feed_forward(dim_model, dim_feedforward),
#            dimension=dim_model,
#            dropout=dropout,
#        )
#
#    def forward(self, src: Tensor) -> Tensor:
#        src = self.attention(src, src, src)
#        return self.feed_forward(src)
#
#
#class TransformerEncoder(nn.Module):
#    def __init__(
#        self,
#        num_layers: int = 6,
#        dim_model: int = 512,
#        num_heads: int = 8,
#        dim_feedforward: int = 2048,
#        dropout: float = 0.1,
#    ):
#        super().__init__()
#        self.layers = nn.ModuleList(
#            [
#                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
#                for _ in range(num_layers)
#            ]
#        )
#
#    def forward(self, src: Tensor) -> Tensor:
#        seq_len, dimension = src.size(1), src.size(2)
#        src += position_encoding(seq_len, dimension)
#        for layer in self.layers:
#            src = layer(src)
#
#        return src
#
#class TransformerDecoderLayer(nn.Module):
#    def __init__(
#        self,
#        dim_model: int = 512,
#        num_heads: int = 6,
#        dim_feedforward: int = 2048,
#        dropout: float = 0.1,
#    ):
#        super().__init__()
#        dim_q = dim_k = max(dim_model // num_heads, 1)
#        self.attention_1 = Residual(
#            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
#            dimension=dim_model,
#            dropout=dropout,
#        )
#        self.attention_2 = Residual(
#            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
#            dimension=dim_model,
#            dropout=dropout,
#        )
#        self.feed_forward = Residual(
#            feed_forward(dim_model, dim_feedforward),
#            dimension=dim_model,
#            dropout=dropout,
#        )
#
#    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
#        tgt = self.attention_1(tgt, tgt, tgt)
#        tgt = self.attention_2(tgt, memory, memory)
#        return self.feed_forward(tgt)
#
#
#class TransformerDecoder(nn.Module):
#    def __init__(
#        self,
#        num_layers: int = 6,
#        dim_model: int = 512,
#        num_heads: int = 8,
#        dim_feedforward: int = 2048,
#        dropout: float = 0.1,
#    ):
#        super().__init__()
#        self.layers = nn.ModuleList(
#            [
#                TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
#                for _ in range(num_layers)
#            ]
#        )
#        self.linear = nn.Linear(dim_model, dim_model)
#
#    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
#        seq_len, dimension = tgt.size(1), tgt.size(2)
#        tgt += position_encoding(seq_len, dimension)
#        for layer in self.layers:
#            tgt = layer(tgt, memory)
#
#        return torch.softmax(self.linear(tgt), dim=-1)
#
#class Transformer(nn.Module):
#    def __init__(
#        self,
#        num_encoder_layers: int = 6,
#        num_decoder_layers: int = 6,
#        dim_model: int = 512,
#        num_heads: int = 6,
#        dim_feedforward: int = 2048,
#        dropout: float = 0.1,
#        activation: nn.Module = nn.ReLU(),
#    ):
#        super().__init__()
#        self.encoder = TransformerEncoder(
#            num_layers=num_encoder_layers,
#            dim_model=dim_model,
#            num_heads=num_heads,
#            dim_feedforward=dim_feedforward,
#            dropout=dropout,
#        )
#        self.decoder = TransformerDecoder(
#            num_layers=num_decoder_layers,
#            dim_model=dim_model,
#            num_heads=num_heads,
#            dim_feedforward=dim_feedforward,
#            dropout=dropout,
#        )
#
#    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
#        return self.decoder(tgt, self.encoder(src))
#
##class PKSpell_single(nn.Module):
##    """Vanilla RNN Model used for comparison. Only a single RNN is used."""
##
##    def __init__(
##        self,
##        input_dim=17,
##        hidden_dim=300,
##        pitch_to_ix=pitch_to_ix,
##        ks_to_ix=ks_to_ix,
##        rnn_depth=1,
#        cell_type="GRU",
#        dropout=None,
#        bidirectional=True,
#        mode="both",
#    ):
#        super(PKSpell_single, self).__init__()
#
#        self.n_out_pitch = len(pitch_to_ix)
#        self.n_out_ks = len(ks_to_ix)
#        if hidden_dim % 2 != 0:
#            raise ValueError("Hidden_dim must be an even integer")
#        self.hidden_dim = hidden_dim
#
#        if cell_type == "GRU":
#            rnn_cell = nn.GRU
#        elif cell_type == "LSTM":
#            rnn_cell = nn.LSTM
#        else:
#            raise ValueError(f"Unknown RNN cell type: {cell_type}")
#
#        # RNN layer.
#        self.rnn = rnn_cell(
#            input_size=input_dim,
#            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
#            bidirectional=bidirectional,
#            num_layers=rnn_depth,
#        )
#
#        if dropout is not None and dropout > 0:
#            self.dropout = nn.Dropout(p=dropout)
#        else:
#            self.dropout = None
#
#        # Output layers. The input will be two times
#        self.top_layer_pitch = nn.Linear(hidden_dim, self.n_out_pitch)
#        self.top_layer_ks = nn.Linear(hidden_dim, self.n_out_ks)
#
#        # Loss function that we will use during training.
#        self.loss_pitch = nn.CrossEntropyLoss(
#            reduction="mean", ignore_index=pitch_to_ix[PAD]
#        )
#        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])
#        self.mode = mode
#
#    def compute_outputs(self, sentences, sentences_len):
#        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
#        rnn_out, _ = self.rnn(sentences)
#        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)
#
#        if self.dropout is not None:
#            rnn_out = self.dropout(rnn_out)
#
#        out_pitch = self.top_layer_pitch(rnn_out)
#        out_ks = self.top_layer_ks(rnn_out)
#
#        return out_pitch, out_ks
#
#    def forward(self, sentences, pitches, keysignatures, sentences_len):
#        # First computes the predictions, and then the loss function.
#
#        # Compute the outputs. The shape is (max_len, n_sentences, n_labels).
#        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)
#
#        # Flatten the outputs and the gold-standard labels, to compute the loss.
#        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
#        scores_pitch = scores_pitch.view(-1, self.n_out_pitch)
#        scores_ks = scores_ks.view(-1, self.n_out_ks)
#        pitches = pitches.view(-1)
#        keysignatures = keysignatures.view(-1)
#        if self.mode == "both":
#            loss = self.loss_pitch(scores_pitch, pitches) + self.loss_ks(
#                scores_ks, keysignatures
#            )
#        elif self.mode == "ks":
#            loss = self.loss_ks(scores_ks, keysignatures)
#        elif self.mode == "ps":
#            loss = self.loss_pitch(scores_pitch, pitches)
#        return loss
#
#    def predict(self, sentences, sentences_len):
#        # Compute the outputs from the linear units.
#        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)
#
#        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
#        predicted_pitch = scores_pitch.argmax(dim=2)
#        predicted_ks = scores_ks.argmax(dim=2)
#        return (
#            [
#                predicted_pitch[: int(l), i].cpu().numpy()
#                for i, l in enumerate(sentences_len)
#            ],
#            [
#                predicted_ks[: int(l), i].cpu().numpy()
#                for i, l in enumerate(sentences_len)
#            ],
#        )
#
#
#class PKSpell(nn.Module):
#    """Models that decouples key signature estimation from pitch spelling by adding a second RNN.
#
#    This model reached state of the art performances for pitch spelling.
#    """
#
#    def __init__(
#        self,
#        input_dim=17,
#        hidden_dim=300,
#        pitch_to_ix=pitch_to_ix,
#        ks_to_ix=ks_to_ix,
#        hidden_dim2=24,
#        rnn_depth=1,
#        dropout=None,
#        dropout2=None,
#        cell_type="GRU",
#        bidirectional=True,
#        mode="both",
#    ):
#        super(PKSpell, self).__init__()
#
#        self.n_out_pitch = len(pitch_to_ix)
#        self.n_out_ks = len(ks_to_ix)
#
#        if hidden_dim % 2 != 0:
#            raise ValueError("Hidden_dim must be an even integer")
#        if hidden_dim2 % 2 != 0:
#            raise ValueError("Hidden_dim2 must be an even integer")
#        self.hidden_dim = hidden_dim
#        self.hidden_dim2 = hidden_dim2
#
#        if cell_type == "GRU":
#            rnn_cell = nn.GRU
#        elif cell_type == "LSTM":
#            rnn_cell = nn.LSTM
#        else:
#            raise ValueError(f"Unknown RNN cell type: {cell_type}")
#
#        # RNN layer.
#        self.rnn = rnn_cell(
#            input_size=input_dim,
#            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
#            bidirectional=bidirectional,
#            num_layers=rnn_depth,
#        )
#        self.rnn2 = rnn_cell(
#            input_size=hidden_dim,
#            hidden_size=hidden_dim2 // 2 if bidirectional else hidden_dim2,
#            bidirectional=bidirectional,
#            num_layers=rnn_depth,
#        )
#
#        if dropout is not None and dropout > 0:
#            self.dropout = nn.Dropout(p=dropout)
#        else:
#            self.dropout = None
#        if dropout2 is not None and dropout2 > 0:
#            self.dropout2 = nn.Dropout(p=dropout2)
#        else:
#            self.dropout2 = None
#
#        # Output layers.
#        self.top_layer_pitch = nn.Linear(hidden_dim, self.n_out_pitch)
#        self.top_layer_ks = nn.Linear(hidden_dim2, self.n_out_ks)
#
#        # Loss function that we will use during training.
#        self.loss_pitch = nn.CrossEntropyLoss(
#            reduction="mean", ignore_index=pitch_to_ix[PAD]
#        )
#        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])
#        self.mode = mode
#
#    def compute_outputs(self, sentences, sentences_len):
#        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
#        rnn_out, _ = self.rnn(sentences)
#        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)
#
#        if self.dropout is not None:
#            rnn_out = self.dropout(rnn_out)
#        out_pitch = self.top_layer_pitch(rnn_out)
#
#        # pass the ks information into the second rnn
#        rnn_out = nn.utils.rnn.pack_padded_sequence(rnn_out, sentences_len)
#        rnn_out, _ = self.rnn2(rnn_out)
#        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)
#
#        if self.dropout2 is not None:
#            rnn_out = self.dropout2(rnn_out)
#        out_ks = self.top_layer_ks(rnn_out)
#
#        return out_pitch, out_ks
#
#    def forward(self, sentences, pitches, keysignatures, sentences_len):
#        # First computes the predictions, and then the loss function.
#
#        # Compute the outputs. The shape is (max_len, n_sentences, n_labels).
#        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)
#
#        # Flatten the outputs and the gold-standard labels, to compute the loss.
#        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
#        scores_pitch = scores_pitch.view(-1, self.n_out_pitch)
#        scores_ks = scores_ks.view(-1, self.n_out_ks)
#        pitches = pitches.view(-1)
#        keysignatures = keysignatures.view(-1)
#        if self.mode == "both":
#            loss = self.loss_pitch(scores_pitch, pitches) + self.loss_ks(
#                scores_ks, keysignatures
#            )
#        elif self.mode == "ks":
#            loss = self.loss_ks(scores_ks, keysignatures)
#        elif self.mode == "ps":
#            loss = self.loss_pitch(scores_pitch, pitches)
#        return loss
#
#    def predict(self, sentences, sentences_len):
#        # Compute the outputs from the linear units.
#        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)
#
#        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
#        predicted_pitch = scores_pitch.argmax(dim=2)
#        predicted_ks = scores_ks.argmax(dim=2)
#        return (
#            [
#                predicted_pitch[: int(l), i].cpu().numpy()
#                for i, l in enumerate(sentences_len)
#            ],
#            [
#                predicted_ks[: int(l), i].cpu().numpy()
#                for i, l in enumerate(sentences_len)
#            ],
#        )
#
#
