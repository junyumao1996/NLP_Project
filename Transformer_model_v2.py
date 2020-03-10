import math
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import itertools
import operator
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from collections import Counter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import gensim
import gensim.downloader as api
import urllib.request
import os

class EncoderCNN(nn.Module):
    def __init__(self, target_size):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(resnet.fc.in_features, target_size)
        self.bn = nn.BatchNorm1d(target_size, momentum=0.01)
        self.init_weights()

    def get_params(self):
        return list(self.linear.parameters()) + list(self.bn.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features

class EncoderTransformer(nn.Module):
    def __init__(self, image_feature_size, nhead, n_layers, dropout):
        super(EncoderTransformer, self).__init__()
        self.image_feature_size = image_feature_size
        # define encoder layers
        encoder_layers = TransformerEncoderLayer(image_feature_size, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        # positional encoding
        self.pos_encoder = PositionalEncoding(image_feature_size, dropout)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def get_params(self):
        return list(self.parameters())

    def forward(self, src):
        src *= math.sqrt(self.image_feature_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class EncoderStory2(nn.Module):
    '''
    Fully transfromer-based encoder. 
    '''
    def __init__(self, img_feature_size, nhead, n_layers, dropout=0.5):
        super(EncoderStory2, self).__init__()
        self.cnn = EncoderCNN(img_feature_size)
        self.transformer_Encoder = EncoderTransformer(img_feature_size, nhead, n_layers, dropout)

    def get_params(self):
        return self.cnn.get_params() + self.transformer_Encoder.get_params()

    def forward(self, story_images):
        data_size = story_images.size()
        local_cnn = self.cnn(story_images.view(-1, data_size[2], data_size[3], data_size[4]))
        memory_output = self.transformer_Encoder(local_cnn)
        return output


class DecoderStory(nn.Module):
    def __init__(self, embed_size, nhead, n_layers, hidden_size, vocab, dropout=0.5, pretrain_embed=False):
        super(DecoderStory, self).__init__()

        self.embed_size = embed_size
        self.linear = nn.Linear(hidden_size * 2, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.transformer = DecoderTransformer(embed_size, nhead, n_layers, vocab, dropout, pretrain_embed)
        self.init_weights()

    def get_params(self):
        return list(self.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, story_feature, captions, lengths):
        story_feature = self.linear(story_feature)
        story_feature = self.dropout(story_feature)
        story_feature = F.relu(story_feature)
        result = self.transformer(story_feature, captions, lengths)
        return result

    def inference(self, story_feature):
        story_feature = self.linear(story_feature)
        story_feature = F.relu(story_feature)
        result = self.transformer.inference(story_feature)
        return result


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x : (seq, batch, embed)
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def load_pretrained_embed(vocab, embed_size, wv_model):
    """
    Transfer pre-trained embedding model to current vocabulary.
    """
    vocab_size = len(vocab)
    pre_matrix = np.zeros((vocab_size, embed_size))
    for token in vocab.word2idx.keys():
        idx = vocab.word2idx[token]
        if token in wv_model.vocab:
            pre_matrix[idx, :] = wv_model[token]
    return torch.FloatTensor(pre_matrix)
