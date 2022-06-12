from xml.etree.ElementInclude import default_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import numpy as np
import pkuseg
import argparse
from tqdm import trange,tqdm
import os
from utils import read_corpus,batch_iter
from vocab import Vocab
from model import CNN
import math
from sklearn.metrics import f1_score

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print(" import error ..")
    
from transformers import AdamW,get_linear_schedule_with_warmup

def set_seed():
    # 设置CPU生成随机数的种子，方便下次复现实验结果。
    # 为什么这里随机要设置成3344
    random.seed(3344)
    np.random.seed(3344)
    torch.manual_seed(3344)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(3344)

def tokenizer(text):
    '''
    定义TEXT的tokenize规则
    '''
    # regex = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9]')
    # text = regex.sub(' ', text)
    seg = pkuseg.pkuseg()
    return [word for word in seg.cut(text) if word.strip()]

def train(args,model,train_data,dev_data,vocab,dtype='CNN'):
    LOG_FILE = args.output_file
    with open(LOG_FILE,'a') as fout:
        fout.write('\n')
        fout.write('========'*6)
        fout.write('start training: {}'.format(dtype))
        fout.write('\n')

    time_start = time.time()
    if not os.path.exists(os.path.join('./runs',dtype)):
        os.makedirs(os.path.join('./runs',dtype))
    tb_writer = SummaryWriter(os.path.join('./runs',dtype))

    t_total = args.num_epoch * (math.ceil(len(train_data) / args.batch_size))
    # eps 是一个防止分母为0的小数
    optimizer = AdamW(model.parameters(),lr = args.learning_rate, eps = 1e-8)
    # 学习率预热 学习率从 0 线性增加到预设的 lr
    scheduler = get_linear_schedule_with_warmup(optimizer = optimizer,  num_warmup_steps = args.warmup_steps, num_training_steps = t_total)
    
    criterion = nn.CrossEntropyLoss()
    global_step = 0.
    total_loss = 0.
    logg_loss = 0.
    val_acces = []
    print(args.num_epoch)
    train_epoch = trange(args.num_epoch, desc = 'train_epoch')
    for epoch in train_epoch:
        model.train()

        for src_sents, labels in batch_iter(train_data, args.batch_size, shuffle = True):
            src_sents = vocab.vocab.to_input_tensor(src_sents,args.device)
            global_step += 1
            optimizer.zero_grad()

            logits = model(src_sents)
            y_labels = torch.tensor(labels, device = args.device)

            example_losses = criterion(logits, y_labels)
            
            example_losses.backward()
            # 梯度裁剪 解决梯度爆炸问题
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            total_loss += example_losses.item()
            if global_step % 100 == 0:
                loss_scalar = (total_loss - logg_loss) / 100
                logg_loss = total_loss

                with open(LOG_FILE, 'a') as fout:
                    fout.write('epoch: {}, iter: {}, loss: {}, learn_rate: {}\n'.format(epoch, global_step, loss_scalar, scheduler.get_lr()[0]))

                print('epoch: {}, iter: {}, loss: {}, learn_rate: {}\n'.format(epoch, global_step, loss_scalar, scheduler.get_lr()[0]))
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", loss_scalar, global_step)
            
            print("Epoch: " , epoch, " Training loss: ", total_loss / global_step)

            eval_loss , eval_result = evaluate(args, criterion, model, dev_data, vocab)
            with open(LOG_FILE, 'a') as fout:
                fout.write('EVALUATE: epoch : {}, loss: {}, eval_result: {}\n'.format(epoch, eval_loss, eval_result))
            
            eval_acc = eval_result['acc']
            if len(val_acces) == 0 or eval_acc > max(val_acces):
                print('best model on epoch: {}, eval_acc: {}'.format(epoch, eval_acc))
                torch.save(model.state_dict(), "classfia-best-{}.th".format(dtype))
                val_acces.append(eval_acc)
        time_end = time.time()
        print("run model of {}, taking total {} m".format(dtype,(time_end - time_start)/60))
        with open(LOG_FILE, 'a') as fout:
            fout.write("run model of {}, taking total {} m\n".format(dtype,(time_end - time_start)/60))
            
def evaluate(args, criterion, model, dev_data, vocab):
    model.eval()
    total_loss = 0.
    total_step = 0.
    preds = None
    out_label_ids = None
    with torch.no_grad():
        for src_sents, labels in batch_iter(dev_data, args.batch_size):
            src_sents = vocab.vocab.to_input_tensor(src_sents, args.device)
            logits = model(src_sents)
            labels = torch.tensor(labels, device = args.device)
            example_losses = criterion(logits, labels)

            total_loss += example_losses.item()
            total_step += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis = 0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis = 0)
        # axis = 0 列, axis = 1 行
        preds = np.argmax(preds, axis = 1)
        result = acc_and_f1(preds, out_label_ids)
        model.train()
        print("Evaluation loss ", total_loss/ total_step)
        print("Evaluation result ", result)
        return total_loss / total_step, result


def acc_and_f1(preds, labels):
    # acc 和 f1 还可以加权的？
    acc = (preds == labels).mean()
    f1 = f1_score(y_true = labels, y_pred = preds, average = 'weighted')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def build_vocab(args):
    if not os.path.exists(args.vocab_path):
        src_sents, labels = read_corpus(args.train_data_dir)
        labels = {label: idx for idx, label in enumerate(labels)}
        vocab = Vocab.build(src_sents, labels, args.max_vocab_size, args.min_freq)
        vocab.save(args.vocab_path)
    else:
        vocab = Vocab.load(args.vocab_path)
    return vocab

def main():
    parse = argparse.ArgumentParser()

    parse.add_argument('--train_data_dir', default = 'NLP-beginner-for-lys/task2/data/cnews/cnews.train.txt', type = str, required = False)
    parse.add_argument('--dev_data_dir', default = 'NLP-beginner-for-lys/task2/data/cnews/cnews.val.txt', type = str, required = False)
    parse.add_argument('--test_data_dir', default = 'NLP-beginner-for-lys/task2/data/cnews/cnews.test.txt', type = str, required = False)
    parse.add_argument('--output_file', default = 'deep_model.log', type = str, required = False)
    parse.add_argument('--batch_size', default = 8, type = int)
    parse.add_argument('--do_train', default = True, action = 'store_true', help = 'Whether to run training.')
    parse.add_argument('--do_test', default = True, action = 'store_true', help = 'Whether to run testing.')
    parse.add_argument('--learning_rate', default = 5e-4, type = float)
    parse.add_argument('--num_epoch', default = 10, type = int)
    parse.add_argument('--max_vocab_size', default = 50000, type = int)
    parse.add_argument('--min_freq', default = 2, type = int)
    parse.add_argument('--embed_size', default = 300, type = int)
    parse.add_argument('--hidden_size', default=256, type = int)
    parse.add_argument('--dropout_rate', default=0.2, type = float)
    parse.add_argument('--warmup_steps', default=0, type = int, help = 'Linear warmup over warmup_steps')
    parse.add_argument('--GRAD_CLIP', default=1, type = float)
    parse.add_argument('--vocab_path', default='./vocab.json', type = str)
    parse.add_argument('--do_cnn', default=True, action = 'store_true', help = 'Whether to run training.')
    parse.add_argument('--do_rnn', default=True, action = 'store_true', help = 'Whether to run training.')
    parse.add_argument('--do_avg', default=True, action = 'store_true', help = 'Whether to run training.')
    parse.add_argument('--num_filter', default=100, type = int, help = 'CNN 模型一个 filter 的输出 channels.')

    args = parse.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed()

    if os.path.exists('NLP-beginner-for-lys/task2/data/cnews/cache_train_data'):
        train_data = torch.load('NLP-beginner-for-lys/task2/data/cnews/cache_train_data')
    else:
        train_data = read_corpus(args.train_data_dir)
        train_data = [(text, labs) for text, labs in zip(*train_data)]
        torch.save(train_data, 'NLP-beginner-for-lys/task2/data/cnews/cache_train_data')
    
    if os.path.exists('NLP-beginner-for-lys/task2/data/cnews/cache_dev_data'):
        dev_data = torch.load('NLP-beginner-for-lys/task2/data/cnews/cache_dev_data')
    else:
        dev_data = read_corpus(args.dev_data_dir)
        dev_data = [(text, labs) for text, labs in zip(*dev_data)]
        torch.save(dev_data, 'NLP-beginner-for-lys/task2/data/cnews/cache_dev_data')

    vocab = build_vocab(args)
    label_map = vocab.labels
    print(label_map)

    if args.do_train:
        if args.do_cnn:
            cnn_model = CNN(len(vocab.vocab), args.embed_size, args.num_filter, [2,3,4], len(label_map), dropout = args.dropout_rate)
            cnn_model.to(device)
            train(args, cnn_model, train_data, dev_data, vocab, dtype = 'CNN')
        
    if args.do_test:
        
        if os.path.exists('NLP-beginner-for-lys/task2/data/cnews/cache_test_data'):
            test_data = torch.load('NLP-beginner-for-lys/task2/data/cnews/cache_test_data')
        else:
            test_data = read_corpus(args.test_data_dir)
            test_data = [(text, labs) for text, labs in zip(*test_data)]
            torch.save(test_data,'NLP-beginner-for-lys/task2/data/cnews/cache_test_data')
    
    criterion = nn.CrossEntropyLoss()

    cnn_model = CNN(len(vocab.vocab), args.embed_size, args.num_filter, [2,3,4], len(label_map), dropout=args.dropout_rate)
    cnn_model.load_state_dict(torch.load('classifa-best-CNN.th'))
    cnn_model.to(device)
    cnn_test_loss, cnn_result = evaluate(args, criterion, cnn_model, test_data, vocab)

    with open(args.output_file, 'a') as fout:
        fout.write('\n')
        fout.write('============ test result ============\n')
        fout.write('test model of {}, loss: {}, result: {}\n'.format('CNN',cnn_test_loss, cnn_result))

if __name__ == '__main__':
    main()