from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from transformers import CLIPTokenizer

def prepare_data(data):
    imgs, captions, captions_lens, class_ids, clip_caps, clip_cap_lens, keys, wrong_caps, wrong_caps_len, wrong_cls_id, wrong_clip_caps, wrong_clip_cap_lens = data
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(clip_cap_lens, 0, True)
    sorted_cap_len, _ = \
        torch.sort(captions_lens, 0, True)
    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))
    
    clpcaps = clip_caps["input_ids"][sorted_cap_indices].squeeze()
    attn = clip_caps["attention_mask"][sorted_cap_indices].squeeze()
    clip_caps = {'input_ids':clpcaps,'attention_mask':attn}
    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    keys = [keys[i] for i in sorted_cap_indices.numpy()]

    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
        sorted_cap_len = Variable(sorted_cap_len).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
        sorted_cap_len = Variable(sorted_cap_len)


    w_sorted_cap_lens, w_sorted_cap_indices = \
        torch.sort(wrong_clip_cap_lens, 0, True)
    w_sorted_cap_len, _ = \
        torch.sort(wrong_caps_len, 0, True)

    wclpcaps = wrong_clip_caps["input_ids"][sorted_cap_indices].squeeze()
    wattn = wrong_clip_caps["attention_mask"][sorted_cap_indices].squeeze()
    wrong_clip_caps = {'input_ids':wclpcaps,'attention_mask':wattn}
    wrong_caps = wrong_caps[w_sorted_cap_indices].squeeze()
    wrong_cls_id = wrong_cls_id[w_sorted_cap_indices].numpy()

    if cfg.CUDA:
        wrong_caps = Variable(wrong_caps).cuda()
        w_sorted_cap_lens = Variable(w_sorted_cap_lens).cuda()
        w_sorted_cap_len = Variable(w_sorted_cap_len).cuda()
    else:
        wrong_caps = Variable(wrong_caps)
        w_sorted_cap_lens = Variable(w_sorted_cap_lens)
        w_sorted_cap_len = Variable(w_sorted_cap_len)

    return [real_imgs, captions, sorted_cap_len,
            class_ids, clip_caps,sorted_cap_lens, keys, wrong_caps, w_sorted_cap_len, wrong_cls_id, wrong_clip_caps,w_sorted_cap_lens]


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = torchvision.transforms.Resize(imsize[i])(img)
                
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)
        self.clip_inputs = self.load_text_data_clip(data_dir, split)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions
    
    def cliptokenizer(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    all_captions.append(cap)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
            
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        inputs = tokenizer(all_captions, padding=True, return_tensors="pt")
        return inputs

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1
        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                print("filepath", filepath)
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_text_data_clip(self, data_dir, split):
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        train_captions = self.cliptokenizer(data_dir, train_names)
        test_captions = self.cliptokenizer(data_dir, test_names)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            clip_inputs = train_captions
            filenames = train_names
        else:  # split=='test'
            clip_inputs = test_captions
            filenames = test_names
        return clip_inputs

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='latin1')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def get_clip_caption(self,sent_ix):
            # a list of indices for a sentence
            sent_caption = {"input_ids":self.clip_inputs.input_ids[[sent_ix]],"attention_mask":self.clip_inputs.attention_mask[[sent_ix]]}
            if (sent_caption["input_ids"] == 0).sum() > 0:
                print('ERROR: do not need END (0) token', sent_caption["input_ids"])
            cont = 0
            for j in sent_caption["attention_mask"][0]:
                if j != 0:
                    cont+=1
                else:
                    break
            num_words = cont
            x = np.full((1, cfg.TEXT.WORDS_NUM), 49407,dtype='int64')
            t = np.zeros((1, cfg.TEXT.WORDS_NUM),dtype='int64')
            x = torch.from_numpy(x)
            t = torch.from_numpy(t)
            cx_len = num_words
            if num_words <= cfg.TEXT.WORDS_NUM:
                for i in range(0,num_words):
                    x[0][i] = sent_caption["input_ids"][0][i]
                    t[0][i] = sent_caption["attention_mask"][0][i]
            else:
                ix = list(np.arange(1,num_words-1))  # 1, 2, 3,..., maxNum
                np.random.shuffle(ix)
                ix = ix[:cfg.TEXT.WORDS_NUM-2]
                ix = np.sort(ix)
                x[0][0] = 49406
                x[0][1:cfg.TEXT.WORDS_NUM-1] = sent_caption["input_ids"][0][ix]
                t[0][0] = 1
                t[0][-1] = 1
                t[0][1:cfg.TEXT.WORDS_NUM-1] = sent_caption["attention_mask"][0][ix]
                cx_len = cfg.TEXT.WORDS_NUM
            sent_clip_caption = {"input_ids":x,"attention_mask":t}
            return sent_clip_caption ,cx_len

    def __getitem__(self, index):

        key = self.filenames[index]
        cls_id = self.class_id[index]

        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir

        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        # randomly select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)
        clip_caps,clip_cap_len = self.get_clip_caption(new_sent_ix)

        # randomly select a mismatch sentence
        wrong_idx = random.randint(0, len(self.filenames))
        wrong_new_sent_ix = wrong_idx * self.embeddings_num + sent_ix
        wrong_caps, wrong_cap_len = self.get_caption(wrong_new_sent_ix)
        wrong_clip_caps,wrong_clip_cap_len = self.get_clip_caption(wrong_new_sent_ix)
        wrong_cls_id = self.class_id[wrong_idx]

        return imgs, caps, cap_len, cls_id, clip_caps, clip_cap_len, key, wrong_caps, wrong_cap_len, wrong_cls_id, wrong_clip_caps, wrong_clip_cap_len

    def get_mis_caption(self, cls_id):
        clpcaps = []
        attn = []
        mis_match_captions = {}
        
        while len(clpcaps) < 99:
            idx = random.randint(0, self.number_example)
            if cls_id == self.class_id[idx]:
                continue
            sent_ix = random.randint(0, self.embeddings_num)
            new_sent_ix = idx * self.embeddings_num + sent_ix
            clip_caps_t,_= self.get_clip_caption(new_sent_ix)
            clpcap = clip_caps_t["input_ids"].numpy()
            clpcaps.append(clpcap)
            clpcapattn = clip_caps_t["attention_mask"].numpy()
            attn.append(clpcapattn)

        clpcaps = np.array(clpcaps)
        clpcaps = torch.from_numpy(clpcaps).squeeze()
        attn = np.array(attn)
        attn = torch.from_numpy(attn).squeeze()
        mis_match_captions = {'input_ids':clpcaps,'attention_mask':attn}
        return mis_match_captions

    def __len__(self):
        return len(self.filenames)

    #for rnn model
    '''def get_mis_caption(self, cls_id):
    mis_match_captions_t = []
    mis_match_captions = torch.zeros(99, cfg.TEXT.WORDS_NUM)
    mis_match_captions_len = torch.zeros(99)
    i = 0
    while len(mis_match_captions_t) < 99:
        idx = random.randint(0, self.number_example)
        if cls_id == self.class_id[idx]:
            continue
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = idx * self.embeddings_num + sent_ix
        # caps_t, cap_len_t = self.get_caption(new_sent_ix)
        clip_caps_t, clip_cap_len_t = self.get_clip_caption(new_sent_ix)
        # mis_match_captions_t.append(torch.from_numpy(caps_t).squeeze())
        mis_match_captions_t.append(clip_caps_t)
        # mis_match_captions_len[i] = cap_len_t
	mis_match_captions_len[i] = clip_cap_len_t
        i = i +1
    sorted_cap_lens, sorted_cap_indices = torch.sort(mis_match_captions_len, 0, True)
    #import ipdb
    #ipdb.set_trace()
    for i in range(99):
        mis_match_captions[i,:] = mis_match_captions_t[sorted_cap_indices[i]]
    # return mis_match_captions.type(torch.LongTensor).cuda(), sorted_cap_lens.type(torch.LongTensor).cuda()
    return mis_match_captions, sorted_cap_lens.type(torch.LongTensor).cuda()'''
