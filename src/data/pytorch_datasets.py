import kmeans1d
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

from src.utils.constants import PAD, KEY_SIGNATURES, PITCHES


class PSDataset(Dataset):
    def __init__(
        self,
        list_of_dict_dataset,
        paths,
        transf_pc,
        transf_tpc,
        transf_k,
        augment_dataset=False,
        sort=False,
        truncate=None,
    ):
        if sort:  # sort (in reverse order) to reduce padding
            list_of_dict_dataset = sorted(
                list_of_dict_dataset,
                key=lambda e: (len(e["midi_number"])),
                reverse=True,
            )
        if not augment_dataset:  # remove the transposed pieces
            list_of_dict_dataset = [
                e for e in list_of_dict_dataset if e["transposed_of"] == "P1"
            ]
        # consider only pieces in paths
        list_of_dict_dataset = [
            e for e in list_of_dict_dataset if e["original_path"] in paths
        ]

        # extract the useful data from dataset
        self.pc_sequences = [e["midi_number"] for e in list_of_dict_dataset]
        self.tpc_sequences = [e["pitches"] for e in list_of_dict_dataset]
        self.durations = [e["duration"] for e in list_of_dict_dataset]
        self.ks = [e["key_signatures"] for e in list_of_dict_dataset]
        self.eoM = [e["endOfMeasures"] for e in list_of_dict_dataset]

        # the transformations to apply to data
        self.transf_pc = transf_pc
        self.transf_tpc = transf_tpc
        self.transf_ks = transf_k
        self.transf_eoM = ToTensorLong()

        # truncate and pad
        self.truncate = truncate

    def __len__(self):
        return len(self.pc_sequences)

    def __getitem__(self, idx):
        pc_seq = self.pc_sequences[idx]
        tpc_seq = self.tpc_sequences[idx]
        duration_seq = self.durations[idx]
        ks_seq = self.ks[idx]
        eoM_seq = self.eoM[idx]

        # transform
        pc_seq = self.transf_pc(pc_seq, duration_seq)
        tpc_seq = self.transf_tpc(tpc_seq)
        ks_seq = self.transf_ks(ks_seq)

        eoM_seq = self.transf_eoM(eoM_seq)

        if self.truncate is not None:
            if len(tpc_seq) > self.truncate:
                pc_seq = pc_seq[0 : self.truncate]
                tpc_seq = tpc_seq[0 : self.truncate]
                ks_seq = ks_seq[0 : self.truncate]
                eoM_seq = eoM_seq[0 : self.truncate]

        # sanity check
        assert len(pc_seq) == len(tpc_seq) == len(ks_seq)
        seq_len = len(tpc_seq)

        return pc_seq, tpc_seq, ks_seq, seq_len, eoM_seq


N_DURATION_CLASSES = 4
accepted_pitches = [ii for i in PITCHES.values() for ii in i]
accepted_ks = KEY_SIGNATURES
pitch_to_ix = {p: accepted_pitches.index(p) for p in accepted_pitches}
ks_to_ix = {k: KEY_SIGNATURES.index(k) for k in KEY_SIGNATURES}
# add PADDING TAD
pitch_to_ix[PAD] = len(accepted_pitches)
ks_to_ix[PAD] = len(KEY_SIGNATURES)

midi_to_ix = {m: m for m in range(12)}
# add PADDING TAD
midi_to_ix[PAD] = 12


class Pitch2Int:
    def __call__(self, in_seq):
        idxs = [pitch_to_ix[w] for w in in_seq]
        return idxs


class Ks2Int:
    def __call__(self, in_seq):
        idxs = [ks_to_ix[w] for w in in_seq]
        return idxs


class Int2Pitch:
    def __call__(self, in_seq):
        return [accepted_pitches[i] for i in in_seq]


class OneHotEncoder:
    def __init__(self, alphabet_len):
        self.alphabet_len = alphabet_len

    def __call__(self, sample):
        onehot = np.zeros([len(sample), self.alphabet_len])
        tot_chars = len(sample)
        onehot[np.arange(tot_chars), sample] = 1
        return onehot


class DurationOneHotEncoder:
    def __init__(self, pitch_alphabet_len, n_dur_class=4):
        self.pitch_alphabet_len = pitch_alphabet_len
        self.dur_alphabet_len = n_dur_class

    def __call__(self, sample, durs):
        sample = torch.tensor(sample, dtype=torch.long)
        onehot_pitch = torch.nn.functional.one_hot(sample, self.pitch_alphabet_len)
        # compute breaks in duration list
        clusters, centroids = kmeans1d.cluster(durs, N_DURATION_CLASSES)
        quantized_durations = torch.tensor(clusters, dtype=torch.long)
        onehot_duration = torch.nn.functional.one_hot(
            quantized_durations, self.dur_alphabet_len
        )
        return torch.cat([onehot_pitch, onehot_duration], 1)


class ToTensorFloat:
    def __call__(self, sample, durs=None):
        if type(sample) is torch.Tensor:
            return sample.float()
        else:
            return torch.tensor(sample, dtype=torch.float)


class ToTensorLong:
    def __call__(self, sample):
        if type(sample) is torch.Tensor:
            return sample.long()
        else:
            return torch.tensor(sample, dtype=torch.long)


class MultInputCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample, durs):
        for t in self.transforms:
            sample = t(sample, durs)
        return sample


### Define the preprocessing pipeline
transform_tpc = transforms.Compose([Pitch2Int(), ToTensorLong()])
transform_pc = MultInputCompose(
    [DurationOneHotEncoder(len(midi_to_ix), N_DURATION_CLASSES), ToTensorFloat()]
)
transform_key = transforms.Compose([Ks2Int(), ToTensorLong()])
transform_eoM = transforms.Compose([ToTensorLong()])


def pad_collate(batch):
    (chromatic_seq, diatonic_seq, ks_seq, l, eoM) = zip(*batch)

    pc_seq_pad = pad_sequence(chromatic_seq, padding_value=midi_to_ix[PAD])
    tpc_seq_pad = pad_sequence(diatonic_seq, padding_value=pitch_to_ix[PAD])
    ks_seq_pad = pad_sequence(ks_seq, padding_value=ks_to_ix[PAD])
    eoM_pad = pad_sequence(eoM,padding_value=0)
    # sort the sequences by length
    seq_lengths, perm_idx = torch.Tensor(l).sort(0, descending=True)
    pc_seq_pad = pc_seq_pad[:, perm_idx, :]
    tpc_seq_pad = tpc_seq_pad[:, perm_idx]
    ks_seq_pad = ks_seq_pad[:, perm_idx]
    eoM_pad = eoM_pad[:,perm_idx]

    return pc_seq_pad, tpc_seq_pad, ks_seq_pad, seq_lengths, eoM_pad
