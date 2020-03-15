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

from transformers import TransfoXLConfig
from TransfoXLModel import TransfoXLModel


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

        self.cnn = EncoderCNN(img_feature_size)

    def get_params(self):
        return self.cnn.get_params()

    def forward(self, story_images):
        """
        story_images : (batch, 5, channels, height, width)
        """
        data_size = story_images.size()
        image_features = self.cnn(story_images.view(-1, data_size[2], data_size[3], data_size[4])) # (batch * 5, img_feature_size)
        image_features = image_features.view(data_size[0], data_size[1], -1) # (batch, 5, img_feature_size)

        return image_features, None

class DecoderStory(nn.Module):
    def __init__(self, embed_size, encoder_output_size, hidden_size, n_head, n_layers, mem_len, vocab, config):
        super(DecoderStory, self).__init__()

        vocab_size = len(vocab)

        self.xl_config = TransfoXLConfig(
            vocab_size=vocab_size, d_model=hidden_size, d_embed=embed_size, n_head=n_head, div_val=1, 
            n_layer=n_layers, tgt_len=50, mem_len=mem_len, adaptive=False
        )

        self.embed_size = embed_size
        self.encoder_output_size = encoder_output_size
        self.hidden_size = hidden_size
        self.mem_len = mem_len
        self.n_layers = n_layers
        self.padding_len = mem_len - 1

        self.fuse_linear = nn.Linear(hidden_size + encoder_output_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)
        self.transformer_xl = TransfoXLModel(self.xl_config)

        # define start vector for a sentence
        self.start_vec = torch.zeros([1, vocab_size], dtype=torch.float32)
        self.start_vec[0][1] = 10000
        if torch.cuda.is_available():
            self.start_vec = self.start_vec.cuda()

    def get_params(self):
        return list(self.transformer_xl.parameters()) + list(self.fuse_linear.parameters()) + list(self.classifier.parameters())

    def fuse_memory(self, mems, image_features):
        """
        mems: List of (mem_len, batch=1, d_model), len(mems)=n_layer
        image_features: (encoder_output_size, )

        returns: identical size of mems
        """
        new_mems = []
        for mem in mems:
            new_mem = self.fuse_linear(torch.cat([mem.squeeze(1), image_features.expand(self.mem_len, -1)], 1))
            new_mems.append(new_mem.unsqueeze(1))
        return new_mems

    def forward(self, encoder_features, captions, lengths):
        """
        encoder_features: (5, encoder_output_size)
        captions: (5, padded_seq)
        lengths: List of seq lengths, len(lengths)=5
        """
        mems = [torch.zeros(self.mem_len, 1, self.hidden_size).cuda() for _ in range(self.n_layers)]

        outputs = []

        for i, length in enumerate(lengths):
            # fuse corresponding image features into last mems
            feature = encoder_features[i]
            mems = self.fuse_memory(mems, feature)

            # pad or truncate caption to mem_len - 1
            caption = torch.zeros(self.mem_len - 1, dtype=torch.long).cuda()
            copy_len = min(length - 1, self.mem_len - 1)
            caption[:copy_len] = captions[i][:copy_len]

            xl_outputs = self.transformer_xl(caption.unsqueeze(0), padding_len=self.padding_len, mems=mems)
            last_hidden_states, mems = xl_outputs[:2] # last_hidden_states: (1, mem_len - 1, d_model)
            
            output = self.classifier(last_hidden_states.squeeze(0)) # (mem_len - 1, vocab_size)
            output = torch.cat((self.start_vec, output), 0) # (mem_len, vocab_size)

            outputs.append(output)
        return outputs

    def inference(self, encoder_features):
        """
        encoder_features: (5, encoder_output_size)
        """
        results = []
        vocab = self.vocab
        end_vocab = vocab('<end>')
        forbidden_list = [vocab('<pad>'), vocab('<start>'), vocab('<unk>')]
        termination_list = [vocab('.'), vocab('?'), vocab('!')]
        function_list = [vocab('<end>'), vocab('.'), vocab('?'), vocab('!'), vocab('a'), vocab('an'), vocab('am'),
                         vocab('is'), vocab('was'), vocab('are'), vocab('were'), vocab('do'), vocab('does'),
                         vocab('did')]

        cumulated_word = []
        mems = [torch.zeros(self.mem_len, 1, self.hidden_size).cuda() for _ in range(self.n_layers)]

        for feature in encoder_features:
            # update memory
            mems = self.fuse_memory(mems, feature)

            predicted = torch.tensor([1], dtype=torch.long).cuda()
            # store all the previous outputs for next input
            next_input = [1]
            infer_tgt = torch.tensor(next_input, dtype=torch.long).cuda()

            sampled_ids = [predicted, ]

            count = 0
            prob_sum = 1.0

            for i in range(50):
                # foward through modules
                outputs = self.transformer_xl(infer_tgt.unsqueeze(0), padding_len=None, mems=mems)
                last_hidden_states, mems = outputs[:2]
                outputs = self.classifier(last_hidden_states.squeeze(0))

                if predicted not in termination_list:
                    # we only consider the last output token
                    outputs[-1][end_vocab] = -100.0

                for forbidden in forbidden_list:
                    outputs[-1][forbidden] = -100.0

                cumulated_counter = Counter()
                cumulated_counter.update(cumulated_word)

                prob_res = outputs[-1]
                prob_res = self.softmax(prob_res)
                for word, cnt in cumulated_counter.items():
                    if cnt > 0 and word not in function_list:
                        prob_res[word] = prob_res[word] / (1.0 + cnt * 5.0)
                prob_res = prob_res * (1.0 / prob_res.sum())

                candidate = []
                for i in range(100):
                    index = np.random.choice(prob_res.size()[0], 1, p=prob_res.cpu().detach().numpy())[0]
                    candidate.append(index)

                counter = Counter()
                counter.update(candidate)

                sorted_candidate = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)

                predicted, _ = counter.most_common(1)[0]
                cumulated_word.append(predicted)
                next_input.append(predicted)

                predicted = torch.from_numpy(np.array([predicted])).cuda()
                sampled_ids.append(predicted)

                if predicted == 2:
                    break
                # update input for transformer decoder
                infer_tgt = torch.tensor(next_input, dtype=torch.long).cuda()


            results.append(sampled_ids)

        return results
