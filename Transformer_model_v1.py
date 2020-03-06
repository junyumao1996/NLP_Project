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
from torch.nn import TransformerDecoder, TransformerDecoderLayer


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


class EncoderStory(nn.Module):
    def __init__(self, img_feature_size, hidden_size, n_layers):
        super(EncoderStory, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn = EncoderCNN(img_feature_size)
        self.lstm = nn.LSTM(img_feature_size, hidden_size, n_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.linear = nn.Linear(hidden_size * 2 + img_feature_size, hidden_size * 2)
        self.dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(hidden_size * 2, momentum=0.01)
        self.init_weights()

    def get_params(self):
        return self.cnn.get_params() + list(self.lstm.parameters()) + list(self.linear.parameters()) + list(
            self.bn.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, story_images):
        data_size = story_images.size()
        local_cnn = self.cnn(story_images.view(-1, data_size[2], data_size[3], data_size[4]))
        global_rnn, (hn, cn) = self.lstm(local_cnn.view(data_size[0], data_size[1], -1))
        glocal = torch.cat((local_cnn.view(data_size[0], data_size[1], -1), global_rnn), 2)
        output = self.linear(glocal)
        output = self.dropout(output)
        output = self.bn(output.contiguous().view(-1, self.hidden_size * 2)).view(data_size[0], data_size[1], -1)

        return output, (hn, cn)


class DecoderStory(nn.Module):
    def __init__(self, embed_size, nhead, n_layers, hidden_size, vocab, dropout=0.5):
        super(DecoderStory, self).__init__()

        self.embed_size = embed_size
        self.linear = nn.Linear(hidden_size * 2, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.transformer = DecoderTransformer(embed_size, nhead, n_layers, vocab, dropout)
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


class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, nhead, n_layers, vocab, dropout=0.5):
        super(DecoderTransformer, self).__init__()
        # vocabulary
        self.vocab = vocab
        vocab_size = len(vocab)
        # encoder for embedding and positional encoding
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        # define transformer decoder_layer and decoder
        decoder_layer = nn.TransformerDecoderLayer(embed_size, nhead)
        self.transformer_decoder = TransformerDecoder(decoder_layer, n_layers)
        # FC linear decoder
        self.decoder = nn.Linear(embed_size, vocab_size)
        self.n_layers = n_layers
        self.softmax = nn.Softmax(0)
        self.brobs = []
        # initial input
        self.init_input = torch.zeros([5, 1, embed_size], dtype=torch.float32)

        if torch.cuda.is_available():
            self.init_input = self.init_input.cuda()
        # define start vector for a sentence
        self.start_vec = torch.zeros([1, vocab_size], dtype=torch.float32)
        self.start_vec[0][1] = 10000
        if torch.cuda.is_available():
            self.start_vec = self.start_vec.cuda()

        self.init_weights()

    def get_params(self):
        return list(self.parameters())

    def init_weights(self):
        # initialize weight for the linear encoder and decoder
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        # generate mask
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, features, captions, lengths, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        '''
        features: (5, embed_size)
        '''
        # waiting for further modifications to tgt_mask and memory_mask...
        # tgt
        embeddings = self.encoder(captions)
        embeddings = self.pos_encoder(embeddings)
        # story features are treated as memory
        features = features.unsqueeze(1)

        outputs = []

        for i, length in enumerate(lengths):
            tgt = embeddings[i][0:length - 1]
            memory = features[i]
            output = self.transformer_decoder(tgt.unsqueeze(1), memory.unsqueeze(1))
            output = self.decoder(output.squeeze(1))
            output = torch.cat((self.start_vec, output), 0)
            outputs.append(output)
        return outputs

    def inference(self, features):
        '''
        features: (5, embed_size)
        '''
        results = []
        vocab = self.vocab
        end_vocab = vocab('<end>')
        forbidden_list = [vocab('<pad>'), vocab('<start>'), vocab('<unk>')]
        termination_list = [vocab('.'), vocab('?'), vocab('!')]
        function_list = [vocab('<end>'), vocab('.'), vocab('?'), vocab('!'), vocab('a'), vocab('an'), vocab('am'),
                         vocab('is'), vocab('was'), vocab('are'), vocab('were'), vocab('do'), vocab('does'),
                         vocab('did')]

        cumulated_word = []
        for feature in features:

            feature = feature.unsqueeze(0).unsqueeze(0)
            predicted = torch.tensor([1], dtype=torch.long).cuda()
            # initialize input for transformer decoder
            infer_memory = feature
            infer_tgt = self.pos_encoder(self.encoder(predicted)).unsqueeze(1)

            sampled_ids = [predicted, ]

            count = 0
            prob_sum = 1.0

            for i in range(50):
                outputs = self.transformer_decoder(infer_tgt.unsqueeze(0), infer_memory.unsqueeze(0))
                outputs = self.decoder(outputs.squeeze(1))

                if predicted not in termination_list:
                    outputs[0][end_vocab] = -100.0

                for forbidden in forbidden_list:
                    outputs[0][forbidden] = -100.0

                cumulated_counter = Counter()
                cumulated_counter.update(cumulated_word)

                prob_res = outputs[0]
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

                predicted = torch.from_numpy(np.array([predicted])).cuda()
                sampled_ids.append(predicted)

                if predicted == 2:
                    break
                # update input for transformer decoder
                infer_memory = feature
                infer_tgt = self.pos_encoder(self.encoder(predicted)).unsqueeze(1)


            results.append(sampled_ids)

        return results


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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)