import argparse
import torch
import torch.nn as nn
import numpy as np
import argparse
import pickle
import random
import os
import yaml
import subprocess
from data_loader import get_loader
from torch.autograd import Variable
from torchvision import transforms
from build_vocab import Vocabulary
from Transformer_model_v1 import EncoderStory, DecoderStory
from Transformer_model_v2 import EncoderStory2
from PIL import Image
from data_loader import VistDataset
def transform_image(image, transform=None):
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=224 ,
                    help='size for input images')
parser.add_argument('--sis_path', type=str,
                    default='./data/sis/test.story-in-sequence.json')
parser.add_argument('--model_name', type=str,
                    default='21_1L')
parser.add_argument('--result_path', type=str,
                    default='./result.json')
parser.add_argument('--log_step', type=int , default=10,
                    help='step size for prining log info')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=2)

parser.add_argument('--vocab_path', type=str, default='./models/vocab.pkl',
                    help='path for vocabulary wrapper')
parser.add_argument('--config_path', type=str,
                    default='./config/config.yaml',
                    help='path for configuration file')
parser.add_argument('--pad', dest='pad', action='store_true', default=False, help='use padding')
parser.add_argument('--mem_len', type=int, default=50, help='length of memory used in Transformer-XL')

parser.add_argument('--img_feature_size', type=int , default=1024 ,
                    help='dimension of image feature')
parser.add_argument('--embed_size', type=int , default=256 ,
                    help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int , default=1024 ,
                    help='dimension of lstm hidden states')

args = parser.parse_args()


args = parser.parse_args()


challenge_dir = '../VIST-Challenge-NAACL-2018/'
image_dir = './data/test/'
sis_path = args.sis_path
result_path = args.result_path
# embed_path = './models/embed-' + str(args.model_num) + '.pkl'
encoder_path = '../model_saved/encoder-' + str(args.model_name) + '.pkl'
decoder_path = '../model_saved/decoder-' + str(args.model_name) + '.pkl'


transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])

with open(args.vocab_path, 'rb') as f:
    vocab = pickle.load(f)

with open(args.config_path, 'r') as f:
    config = yaml.load(f)

vist = VistDataset(image_dir, sis_path, vocab)
# data_loader = get_loader(image_dir, sis_path, vocab, transform, args.batch_size, shuffle=False, num_workers=args.num_workers)


###### full transformer ######
encoder = EncoderStory2(args.img_feature_size, 4, 1)
decoder = DecoderStory(args.embed_size, 4, 1, int(args.hidden_size/2), vocab)
###### transfomrer XL ######
# encoder = EncoderStory(args.img_feature_size, config)
# decoder = DecoderStory(args.embed_size, args.img_feature_size, args.hidden_size, 4, 1, args.mem_len, vocab, config)


encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))

encoder.eval()
decoder.eval()
if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()
    print("Cuda is enabled...")


# fetch image and perform inference
# image id
image_id = 1300
images, targets, photo_sequences, album_ids = vist.GetItem(image_id)

image_tensor = []
for image in images:
    image = transform_image(image, transform)
    image_tensor.append(image)

image_tensor = torch.stack(image_tensor).squeeze(1).unsqueeze(0)
if torch.cuda.is_available():
    image_tensor = image_tensor.cuda()

feature, _ = encoder(image_tensor)
print("feature shape:", feature.shape)
inference_results = decoder.inference(feature.squeeze(0))
print(inference_results)

sentences = []
target_sentences = []
descriptions = []

for i, result in enumerate(inference_results):
    words = []
    for word_id in result:
        word = vocab.idx2word[word_id.cpu().item()]
        words.append(word)
        if word == '<end>':
            break
            
    words.remove('<start>')
    try:
        words.remove('<end>')
    except Exception:
        pass
        
    sentences.append(' '.join(words))

print(sentences)
