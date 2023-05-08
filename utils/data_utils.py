# %%
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Dataset
import numpy as np
from collections import defaultdict
import json
import random
import time
import pickle
import joblib

# based on the code from https://github.com/kniter1/TAILOR

"""
CMU-MOSEI info
Train 16326 samples
Val 1871 samples
Test 4659 samples
CMU-MOSEI feature shapes
visual: (60, 35)
audio: (60, 74)
text: GLOVE->(60, 300)
label: (6) -> [happy, sad, anger, surprise, disgust, fear] 
    averaged from 3 annotators
unaligned:
text: (50, 300)
visual: (500, 35)
audio: (500, 74)    
"""

emotion_dict = {4: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5}

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_loader(args):
    if args.dataset == 'Aligned':
        data_path = 'Dataset Path'
        train_set = AlignedMoseiDataset(data_path, 'train')
        dev_set = AlignedMoseiDataset(data_path, 'valid')
        test_set = AlignedMoseiDataset(data_path, 'test')

    elif args.dataset == 'UnAligned':
        data_path = 'Dataset Path'
        label_path = 'Label Path'
        train_set = UnAlignedMoseiDataset(data_path, label_path, 'train')
        dev_set = UnAlignedMoseiDataset(data_path, label_path, 'valid')
        test_set = UnAlignedMoseiDataset(data_path, label_path, 'test')

    elif args.dataset == 'NEMu':
        data_path = 'Dataset Path'
        train_set = NEMuDataset(data_path, 'train')
        dev_set = NEMuDataset(data_path, 'valid')
        test_set = NEMuDataset(data_path, 'test')

    else:
        raise NotImplementedError

    train_sampler = RandomSampler(train_set)
    dev_sampler = SequentialSampler(dev_set)
    test_sampler = SequentialSampler(test_set)

    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=2,
                              pin_memory=True,
                              drop_last=True)
    dev_loader = DataLoader(dev_set,
                            sampler=dev_sampler,
                            batch_size=args.eval_batch_size,
                            num_workers=2,
                            pin_memory=False) if dev_set is not None else None
    test_loader = DataLoader(test_set,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=2,
                             pin_memory=False) if test_set is not None else None

    return train_loader, dev_loader, test_loader


class AlignedMoseiDataset(Dataset):
    def __init__(self, data_path, data_type):
        self.data_path = data_path
        self.data_type = data_type
        self.visual, self.audio, \
        self.text, self.labels = self._get_data(self.data_type)

        print('>>', data_type, ' video feature: ', self.visual.shape)
        print('>>', data_type, ' audio feature: ', self.audio.shape)
        print('>>', data_type, ' text feature: ', self.text.shape)

    def _get_data(self, data_type):
        data = torch.load(self.data_path)

        data = data[data_type]
        visual = data['src-visual']
        audio = data['src-audio']
        text = data['src-text']
        labels = data['tgt']

        return visual, audio, text, labels

    def _get_text(self, index):
        text = torch.FloatTensor(self.text[index])

        return text

    def _get_visual(self, index):
        visual = torch.FloatTensor(self.visual[index])

        return visual

    def _get_audio(self, index):
        audio = self.audio[index]
        audio[audio == -np.inf] = 0
        audio = torch.FloatTensor(audio)

        return audio

    def _get_labels(self, index):
        label_list = self.labels[index]
        label = np.zeros(6, dtype=np.float32)
        filter_label = label_list[1:-1]
        for emo in filter_label:
            label[emotion_dict[emo]] = 1

        return label

    def _get_label_input(self):
        labels_embedding = np.arange(6)
        labels_embedding = torch.from_numpy(labels_embedding)

        return labels_embedding

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self._get_text(index)
        visual = self._get_visual(index)
        audio = self._get_audio(index)
        label = self._get_labels(index)

        return text, visual, audio, label


class UnAlignedMoseiDataset(Dataset):
    def __init__(self, data_path, label_path, data_type):
        self.data_path = data_path
        self.label_path = label_path
        self.data_type = data_type
        self.visual, self.audio, \
        self.text, self.labels = self._get_data(self.data_type)

        print('>>', data_type, ' video feature: ', self.visual.shape)
        print('>>', data_type, ' audio feature: ', self.audio.shape)
        print('>>', data_type, ' text feature: ', self.text.shape)

    def _get_data(self, data_type):
        label_data = torch.load(self.label_path)
        label_data = label_data[data_type]
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        data = data[data_type]
        visual = data['vision']
        audio = data['audio']
        text = data['text']
        audio = np.array(audio)
        labels = label_data['tgt']
        return visual, audio, text, labels

    def _get_text(self, index):
        text = torch.FloatTensor(self.text[index])

        return text

    def _get_visual(self, index):
        visual = torch.FloatTensor(self.visual[index])

        return visual

    def _get_audio(self, index):
        audio = self.audio[index]
        audio[audio == -np.inf] = 0
        audio = torch.FloatTensor(audio)

        return audio

    def _get_labels(self, index):
        label_list = self.labels[index]
        label = np.zeros(6, dtype=np.float32)
        filter_label = label_list[1:-1]
        for emo in filter_label:
            label[emotion_dict[emo]] = 1

        return label

    def _get_label_input(self):
        labels_embedding = np.arange(6)
        labels_embedding = torch.from_numpy(labels_embedding)

        return labels_embedding

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self._get_text(index)
        visual = self._get_visual(index)
        audio = self._get_audio(index)
        label = self._get_labels(index)

        return text, visual, audio, label


class NEMuDataset(Dataset):
    def __init__(self, data_path, data_type):
        self.data_path = data_path
        self.data_type = data_type
        self.img, self.aud, self.com, self.lyr, self.labels = self._get_data(self.data_type)

    def _get_data(self, data_type):
        with open(self.data_path, 'rb') as f:
            data = torch.load(f)
        data = data[data_type]
        img = data['img']
        aud = data['aud']
        com = data['com']
        lyr = data['lyr']
        labels = data['tgt']
        return img, aud, com, lyr, labels

    def _get_com(self, index):
        com = torch.FloatTensor(self.com[index])
        com_mask = torch.BoolTensor([0] * com.shape[0])
        return com, com_mask

    def _get_lyr(self, index):
        lyr = torch.FloatTensor(self.lyr[index])
        lyr_mask = torch.BoolTensor([0] * lyr.shape[0])
        return lyr, lyr_mask

    def _get_img(self, index):
        img = torch.FloatTensor(self.img[index])
        img_mask = torch.BoolTensor([0] * 1)
        return img, img_mask

    def _get_aud(self, index):
        aud = torch.FloatTensor(self.aud[index])
        aud_mask = torch.BoolTensor([0] * aud.shape[0])
        return aud, aud_mask

    def _get_labels(self, index):
        label = torch.LongTensor(self.labels[index])
        return label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        com, com_mask = self._get_com(index)
        lyr, lyr_mask = self._get_lyr(index)
        img, img_mask = self._get_img(index)
        aud, aud_mask = self._get_aud(index)
        label = self._get_labels(index)

        return com, com_mask, \
               lyr, lyr_mask, \
               img, img_mask, \
               aud, aud_mask, label


class MOSIDataset(Dataset):
    def __init__(self, data_path, data_type):
        self.data_path = data_path
        self.data_type = data_type
        self.visual, self.audio, \
        self.text, self.labels = self._get_data(self.data_type)

        print('>>', data_type, ' video feature: ', self.visual.shape)
        print('>>', data_type, ' audio feature: ', self.audio.shape)
        print('>>', data_type, ' text feature: ', self.text.shape)

    def _get_data(self, data_type):
        data = load_pickle(self.data_path)

        data = data[data_type]
        visual = data['vision']
        audio = data['audio']
        text = data['text']
        labels = data['labels']

        return visual, audio, text, labels

    def _get_text(self, index):
        text = torch.FloatTensor(self.text[index])

        return text

    def _get_visual(self, index):
        visual = torch.FloatTensor(self.visual[index])

        return visual

    def _get_audio(self, index):
        audio = self.audio[index]
        audio[audio == -np.inf] = 0
        audio = torch.FloatTensor(audio)

        return audio

    def _get_labels(self, index):
        label = self.labels[index]
        label = torch.FloatTensor(label).squeeze(-1).squeeze(-1)

        return label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self._get_text(index)
        visual = self._get_visual(index)
        audio = self._get_audio(index)
        label = self._get_labels(index)

        return text, visual, audio, label
