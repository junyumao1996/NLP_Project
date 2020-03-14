import os
import math
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import operator
from collections import Counter

from transformers import TransfoXLModel, TransfoXLConfig


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
        """
        images : (batch * 5, channels, height, width)
        """
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features

class EncoderStory(nn.Module):
    def __init__(self, img_feature_size, config):
        super(EncoderStory, self).__init__()

        self.cnn = EncoderCNN(img_feature_size, config)

    def get_params(self):
        return self.cnn.get_params()

    def forward(self, story_images):
        """
        story_images : (batch, 5, channels, height, width)
        """
        data_size = story_images.size()
        image_features = self.cnn(story_images.view(-1, data_size[2], data_size[3], data_size[4])) # (batch * 5, img_feature_size)
        image_features = image_features.view(data_size[0], data_size[1], -1) # (batch, 5, img_feature_size)

        return image_features

class DecoderStory(nn.Module):
    def __init__(self, embed_size, encoder_output_size, hidden_size, n_head, n_layers, vocab, config):
        super(DecoderStory, self).__init__()

        self.xl_config = TransfoXLConfig(
            vocab_size=len(vocab), d_model=hidden_size, d_embed=embed_size, n_head=n_head, div_val=1, 
            n_layer=n_layers, tgt_len=50, mem_len=500, adaptive=False
        )

        self.embed_size = embed_size
        self.encoder_output_size = encoder_output_size
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(p=0.5)
        self.transformer_xl = TransfoXLModel(self.xl_config)

    def get_params(self):
        return list(self.transformer_xl.parameters())

    def fuse_memory(self, mems, image_features):
        """
        mems: List of (mem_len, batch=1, d_model), len(mems)=n_layer
        image_features: (5, encoder_output_size)

        returns: identical size of mems
        """
        pass

    def forward(self, encoder_features, captions, lengths):
        """
        encoder_features: (5, encoder_output_size)
        captions: (5, padded_seq)
        lengths: List of seq lengths, len(lengths)=5
        """

        pass
    def inference(self, encoder_features):
        """
        encoder_features: (5, encoder_output_size)
        """
        pass