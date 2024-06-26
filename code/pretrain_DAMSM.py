from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data
from dataset_fashiongen2 import TextDataset as TextFashionGenDataset
#from dataset_fashiongen import TextDataset as TextFashionGenDataset

##from model import RNN_ENCODER, CNN_ENCODER
from model import CLIP_TEXT_ENCODER,CLIP_VISION_ENCODER
from transformers import CLIPTextModel

import os
import sys
import traceback
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

encoders_type = cfg.ENCODERS_TYPE
UPDATE_INTERVAL = 2000  # default value was 200 (Stephane)
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def train(dataloader, image_encoder, text_encoder, batch_size,
          labels, optimizer, epoch, ixtoword, image_dir):
          
    image_encoder.train()
    text_encoder.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = (epoch + 1) * len(dataloader)
    start_time = time.time()
    for step, data in enumerate(dataloader, 0):
        text_encoder.zero_grad()
        image_encoder.zero_grad()

        imgs, captions, cap_lens, class_ids, clip_caps, clip_cap_lens, keys, wrong_caps, \
                                wrong_caps_len, wrong_cls_id, wrong_clip_caps, wrong_clip_cap_lens = prepare_data(data)

        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = image_encoder(imgs[-1])
        # --> batch_size x nef x 17*17
        nef, att_sze = words_features.size(1), words_features.size(2)
        # words_features = words_features.view(batch_size, nef, -1)

        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        ## Use captions, cap_lens, hidden as Parameters for rnn model 
        #hidden = text_encoder.init_hidden(batch_size)
        words_emb, sent_emb = text_encoder(clip_caps)
        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 clip_cap_lens, class_ids, batch_size)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data
        #
        loss.backward()
        #
        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in TRANSFORMERs / RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(text_encoder.parameters(),
                                      cfg.TRAIN.TEXT_ENCODERS_GRAD_CLIP)

        optimizer.step()

        if step > 0 and step % UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step


            s_cur_loss0 = s_total_loss0 / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1 / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0 / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1 / UPDATE_INTERVAL

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  's_loss {:5.2f} {:5.2f} | '
                  'w_loss {:5.2f} {:5.2f}'
                  .format(epoch, step, len(dataloader),
                          elapsed * 1000. / UPDATE_INTERVAL,
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            start_time = time.time()
            # attention Maps
            img_set, _ = \
                build_super_images(imgs[-1].cpu(), captions,
                                   ixtoword, attn_maps, att_sze)

            if img_set is not None:
                im = Image.fromarray(img_set)
                
                
                fullpath = '%s/attention_maps%d.png' % (image_dir, step)
                im.save(fullpath)
    return count


def evaluate(dataloader, image_encoder, text_encoder, batch_size):
    image_encoder.eval()
    text_encoder.eval()
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, cap_lens, class_ids, clip_caps, clip_cap_lens, keys, \
                wrong_caps, wrong_caps_len, wrong_cls_id, wrong_clip_caps, wrong_clip_cap_lens = prepare_data(data)

        words_features, sent_code = image_encoder(real_imgs[-1])
        #hidden = text_encoder.init_hidden(batch_size)
        words_emb, sent_emb = text_encoder(clip_caps)

        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                            clip_cap_lens, class_ids, batch_size)

        w_total_loss += (w_loss0 + w_loss1).data

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data

        if step == 50:
            break

    s_cur_loss = s_total_loss/ step
    w_cur_loss = w_total_loss/ step

    return s_cur_loss, w_cur_loss


def build_models():
    # build model ############################################################
    if encoders_type == 'TRANSFORMERS':
        text_encoder = CLIP_TEXT_ENCODER()        
        image_encoder = CLIP_VISION_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    elif encoders_type == 'RNN&CNN':
        text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    else:
        raise NotImplementedError

    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    if cfg.DATASET_NAME == 'fashiongen2':
        image_transform = transforms.Compose([
            transforms.Resize(imsize),
            transforms.RandomHorizontalFlip()])
    else:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])

    if cfg.DATASET_NAME == 'fashiongen2':
        dataset = TextFashionGenDataset(cfg.DATA_DIR, 'train', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    else:
        dataset = TextDataset(cfg.DATA_DIR, 'train', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
        
    print(dataset.n_words, dataset.embeddings_num)
    assert dataset

    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #
    if cfg.DATASET_NAME == 'fashiongen2':
        dataset_val = TextFashionGenDataset(cfg.DATA_DIR, 'test', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    else:
        dataset_val = TextDataset(cfg.DATA_DIR, 'test', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
        
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # Train ##############################################################
    text_encoder, image_encoder, labels, start_epoch = build_models()
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)

    try:
        lr = cfg.TRAIN.ENCODER_LR
        best_s_loss = float('inf')
        best_w_loss = float('inf')
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            count = train(dataloader, image_encoder, text_encoder,
                          batch_size, labels, optimizer, epoch,
                          dataset.ixtoword, image_dir)
            print('-' * 89)
            if len(dataloader_val) > 0:
                s_loss, w_loss = evaluate(dataloader_val, image_encoder,
                                          text_encoder, batch_size)
                print('| end epoch {:3d} | valid loss '
                      '{:5.2f} {:5.2f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, lr))
            print('-' * 89)

            if s_loss <= best_s_loss and w_loss <= best_w_loss:
                best_s_loss = s_loss
                best_w_loss = w_loss
                best_epoch = epoch
                filename = 'best_epoch.txt'
                if os.path.exists(filename):
                    # Overwrite the file
                    with open(filename, 'w') as file:
                        file.write(f"Best epoch: {best_epoch}, Best s_loss: {best_s_loss}, Best w_loss: {best_w_loss}")
                    print("File overwritten.")
                else:
                    # Create a new file
                    with open(filename, 'w') as file:
                        file.write(f"Best epoch: {best_epoch}, Best s_loss: {best_s_loss}, Best w_loss: {best_w_loss}")
                    print("File created.")

                bestmodel = '%s/best_models' % (model_dir)
                if not os.path.isdir(bestmodel):
                    print('Make a new folder: ', bestmodel)
                    mkdir_p(bestmodel)

                torch.save(image_encoder.state_dict(),
                           '%s/best_image_encoder.pth' % (bestmodel))
                torch.save(text_encoder.state_dict(),
                           '%s/best_text_encoder.pth' % (bestmodel))
                print('Save best G/Ds models.')
            print(f"Best epoch: {best_epoch}, Best s_loss: {best_s_loss}, Best w_loss: {best_w_loss}")

            if lr > cfg.TRAIN.ENCODER_LR/10.:
                lr *= 0.98

            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == cfg.TRAIN.MAX_EPOCH):
                torch.save(image_encoder.state_dict(),
                           '%s/image_encoder%d.pth' % (model_dir, epoch))
                torch.save(text_encoder.state_dict(),
                           '%s/text_encoder%d.pth' % (model_dir, epoch))
                print('Save G/Ds models.')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
