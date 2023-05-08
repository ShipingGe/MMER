# coding=utf-8

from __future__ import absolute_import, division, print_function

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
import argparse

import random
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from models import MMERModel
from utils import get_loader, AsymmetricLoss, AsymmetricLossOptimized, FocalLoss_MultiLabel
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
from timm.scheduler.cosine_lr import CosineLRScheduler

def get_accuracy(y, y_pre):
    #	print('metric_acc:  ' + str(round(metrics.accuracy_score(y, y_pre),4)))
    samples = len(y)
    count = 0.0
    for i in range(samples):
        y_true = 0
        all_y = 0
        for j in range(len(y[i])):
            if y[i][j] > 0 and y_pre[i][j] > 0:
                y_true += 1
            if y[i][j] > 0 or y_pre[i][j] > 0:
                all_y += 1
        if all_y <= 0:
            all_y = 1

        count += float(y_true) / float(all_y)
    acc = float(count) / float(samples)
    acc = round(acc, 4)
    return acc


logger = logging.getLogger("__name__")
logging.getLogger().setLevel(logging.INFO)


class APP:
    def __init__(self, model, emb_names):
        self.model = model
        self.emb_names = emb_names
        self.backup = {}

    def perturb(self, epsilon=1.):
        for name, param in self.model.named_parameters():
            for emb_name in self.emb_names:
                if param.requires_grad and emb_name in name:
                    self.backup[name] = param.data.clone()
                    norm = torch.norm(param.grad, keepdim=True)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            for emb_name in self.emb_names:
                if param.requires_grad and emb_name in name:
                    assert name in self.backup
                    param.data = self.backup[name]
        self.backup = {}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(args, model):
    model.eval()
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def load_model(args, model):
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    state_dict = torch.load(model_checkpoint)
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    return model


def load_pretrained_model(args, model):
    model_checkpoint = os.path.join(args.output_dir, "pretrained_model_checkpoint.bin")
    state_dict = torch.load(model_checkpoint)
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    logger.info("Pretrained parameters loaded.")
    return model


def setup(args):
    # model = MMERModel2()
    model = MMERModel(num_encoder_layers=args.n_en,
                      num_decoder_layers=args.n_de,
                      d_model=args.d_model,
                      dim_feedforward=args.dim_feedforward)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model).to(args.device)
    else:
        model.to(args.device)
    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, model, data_loader):
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(data_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []

    epoch_iterator = tqdm(data_loader,
                          desc="Validating... ",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])

    for step, data in enumerate(epoch_iterator):
        text, video, audio, label = data

        text = text.to(args.device)
        video = video.to(args.device)
        audio = audio.to(args.device)

        # com, com_mask, lyr, lyr_mask, img, img_mask, aud, aud_mask, label = data
        #
        # com, com_mask = com.to(args.device), com_mask.to(args.device)
        # lyr, lyr_mask = lyr.to(args.device), lyr_mask.to(args.device)
        # img, img_mask = img.to(args.device), img_mask.to(args.device)
        # aud, aud_mask = aud.to(args.device), aud_mask.to(args.device)

        labels = label.to(args.device)

        with torch.no_grad():
            logits = model(text, video, audio)[0]

        preds = (logits > args.threshold).int().cpu().tolist()
        logits = logits.cpu().tolist()

        all_preds += preds
        all_labels += labels.tolist()
        all_logits += logits

    accuracy = get_accuracy(all_labels, all_preds)
    hl = hamming_loss(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, average='micro')
    r = recall_score(all_labels, all_preds, average='micro')
    mif1 = f1_score(all_labels, all_preds, average='micro')
    maf1 = f1_score(all_labels, all_preds, average='macro')
    #
    print('Accuracy: %.4f' % accuracy)
    print('HammingLoss: %.4f' % hl)
    print('Precision: %.4f' % p)
    print('Recall: %.4f' % r)
    print('Micro F1: %.4f' % mif1)
    print('Macro F1: %.4f' % maf1)

    return accuracy


def train(args, model, train_loader, dev_loader, test_loader):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = CosineLRScheduler(optimizer, t_initial=args.num_epochs, warmup_t=10, warmup_lr_init=5e-6)

    loss_fn = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=1)
    # loss_fn = nn.BCELoss()

    app = APP(model, emb_names=['tri_modal_encoder'])

    logger.info("***** Running training *****")
    logger.info("  Total optimization epochs = %d", args.num_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size / args.n_gpu)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility
    best_dev_f1 = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_iterator = tqdm(train_loader,
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        losses = AverageMeter()
        for step, data in enumerate(epoch_iterator):
            text, video, audio, label = data

            text = text.to(args.device)
            video = video.to(args.device)
            audio = audio.to(args.device)

            # for NEMu dataset, which contains four modalities:
            # com, com_mask, lyr, lyr_mask, img, img_mask, aud, aud_mask, label = data

            # com, com_mask = com.to(args.device), com_mask.to(args.device)
            # lyr, lyr_mask = lyr.to(args.device), lyr_mask.to(args.device)
            # img, img_mask = img.to(args.device), img_mask.to(args.device)
            # aud, aud_mask = aud.to(args.device), aud_mask.to(args.device)

            labels = label.float().to(args.device)

            preds, masked_preds = model(text, video, audio)

            optimizer.zero_grad()

            loss1 = loss_fn(preds, labels)
            loss1.backward(retain_graph=True)

            loss2 = 0.1 * loss_fn(masked_preds, labels)
            loss2.backward(retain_graph=True)

            app.perturb()
            preds_noise, _ = model(text, video, audio)
            loss_noise = F.kl_div(F.log_softmax(preds_noise, dim=-1), F.softmax(preds.clone().detach(), dim=-1))
            loss_noise.backward()
            app.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            losses.update(loss1.item())

            epoch_iterator.set_description(
                "Training (%d / %d Epochs)(cur_loss=%2.4f, avg_loss=%2.4f)" % (
                    epoch + 1, args.num_epochs, losses.val, losses.avg))

        scheduler.step(epoch)


        print('evaluating on dev set......')
        dev_f1 = evaluate(args, model, dev_loader)
        if dev_f1 > best_dev_f1:
            save_model(args, model)
            best_dev_f1 = dev_f1


    # save_model(args, model)

    if args.local_rank in [-1, 0]:
        writer.close()
    # logger.info("mAP: \t%f" % mAP)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset",
                        choices=["Aligned",  "NEMu"],
                        default="Aligned",
                        help="Which downstream task.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=256, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=1e-2, type=float,
                        help="Weight delay if we apply some.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_epochs', type=int, default=10,
                        help="Num of training epochs.")
    parser.add_argument('--eval_only', action='store_true',
                        help='Whether to train or validate the model.')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='the threshold of whether the emotion exists.')
    parser.add_argument('--n_en', default=3, type=int,
                        help='Num of uni- or multi-modal encoders.')
    parser.add_argument('--n_de', default=1, type=int,
                        help='Num of decoders.')
    parser.add_argument('--d_model', default=512, type=int,
                        help=' the number of expected features in the transformer input.')
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help=' the dimension of the feed forward network model.')

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(seconds=30))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))

    set_seed(args)
    args, model = setup(args)
    train_loader, dev_loader, test_loader = get_loader(args)

    if not args.eval_only:
        time_start = time.time()
        train(args, model, train_loader, dev_loader, test_loader)
        time_end = time.time()
        logger.info('Training time cost: %2.1f minutes.' % ((time_end - time_start) / 60))

    # Validating
    model = load_model(args, model)
    # evaluate(args, model, dev_loader)
    evaluate(args, model, test_loader)


if __name__ == "__main__":
    main()
