import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constants import PAD
from src.data.pytorch_datasets import pitch_to_ix, ks_to_ix


class PKSpell_single(nn.Module):
    """Vanilla RNN Model used for comparison. Only a single RNN is used."""

    def __init__(
        self,
        input_dim=17,
        hidden_dim=300,
        pitch_to_ix=pitch_to_ix,
        ks_to_ix=ks_to_ix,
        rnn_depth=1,
        cell_type="GRU",
        dropout=None,
        bidirectional=True,
        mode="both",
    ):
        super(PKSpell_single, self).__init__()

        self.n_out_pitch = len(pitch_to_ix)
        self.n_out_ks = len(ks_to_ix)
        if hidden_dim % 2 != 0:
            raise ValueError("Hidden_dim must be an even integer")
        self.hidden_dim = hidden_dim

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

        # Output layers. The input will be two times
        self.top_layer_pitch = nn.Linear(hidden_dim, self.n_out_pitch)
        self.top_layer_ks = nn.Linear(hidden_dim, self.n_out_ks)

        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])
        self.mode = mode

    def compute_outputs(self, sentences, sentences_len):
        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        rnn_out, _ = self.rnn(sentences)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        if self.dropout is not None:
            rnn_out = self.dropout(rnn_out)

        out_pitch = self.top_layer_pitch(rnn_out)
        out_ks = self.top_layer_ks(rnn_out)

        return out_pitch, out_ks

    def forward(self, sentences, pitches, keysignatures, sentences_len):
        # First computes the predictions, and then the loss function.

        # Compute the outputs. The shape is (max_len, n_sentences, n_labels).
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)

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

    def predict(self, sentences, sentences_len):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)

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


class PKSpell(nn.Module):
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
        super(PKSpell, self).__init__()

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
        self.top_layer_ks = nn.Linear(hidden_dim2, self.n_out_ks)

        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])
        self.mode = mode

    def compute_outputs(self, sentences, sentences_len):
        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        rnn_out, _ = self.rnn(sentences)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        if self.dropout is not None:
            rnn_out = self.dropout(rnn_out)
        out_pitch = self.top_layer_pitch(rnn_out)

        # pass the ks information into the second rnn
        rnn_out = nn.utils.rnn.pack_padded_sequence(rnn_out, sentences_len)
        rnn_out, _ = self.rnn2(rnn_out)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        if self.dropout2 is not None:
            rnn_out = self.dropout2(rnn_out)
        out_ks = self.top_layer_ks(rnn_out)

        return out_pitch, out_ks

    def forward(self, sentences, pitches, keysignatures, sentences_len):
        # First computes the predictions, and then the loss function.

        # Compute the outputs. The shape is (max_len, n_sentences, n_labels).
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)

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

    def predict(self, sentences, sentences_len):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)

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

