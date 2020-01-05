from __future__ import absolute_import
import argparse
import logging
import os
import random
from io import open
import numpy as np
import torch
from torch.utils.data import (DataLoader,TensorDataset)
import time
from tqdm import tqdm
from my_pytorch_transformers.modeling_bert import BertForSequenceClassification, BertConfig
from my_pytorch_transformers import AdamW, WarmupLinearSchedule
from my_pytorch_transformers.tokenization_bert import BertTokenizer
from itertools import cycle
import re
from sklearn.metrics import precision_recall_fscore_support

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None,text_c=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c=text_c
        self.label = label

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label
                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
            # (tokens, input_ids, input_mask, segment_ids) 这是choices_features特征
        ]
        self.label = label

def read_examples(input_file):
    examples = []
    with open(input_file, 'r', encoding='utf-8') as fr:
        for step, line in enumerate(fr):
                raw = line.strip().split("   ")
                if len(raw)==8:
                 user_id = raw[0]
                 news_title = raw[3]
                 news = raw[4]
                 label = int(raw[6])
                 rank=raw[7]
                 examples.append(InputExample(guid=user_id,text_a=news,text_b=news_title,text_c=rank,label=label))
    return examples  # 读取的时候封装以后读出来是一个list,并且同时初始化了InputExample这个类


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    # examples 是list,里面保存的是一个一个的对象。
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in enumerate(examples):  # 按行读取整个文本。
        context_tokens = tokenizer.tokenize(example.text_a)  # 把内容一个一个的句子切割，字级。
        start_tokens = tokenizer.tokenize(example.text_b)  # 把标题切割成一个一个的字级。
        rank_tokens=tokenizer.tokenize(example.text_c)
        _truncate_seq_pair(context_tokens, start_tokens,rank_tokens, max_seq_length - 4)
        choices_features = []
        tokens = ["[CLS]"] + start_tokens +["[SEP]"]+context_tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 此处要求句子长度不能超过512
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids += ([0] * padding_length)
        input_mask += ([0] * padding_length)
        example.guid = re.sub("\D", "", example.guid)  # 除去id中非数字的东西
        segment_ids = int(example.guid)
        label = example.label
        choices_features.append((tokens, input_ids, input_mask, segment_ids))
        features.append(
            InputFeatures(
                example_id=example.guid,
                choices_features=choices_features,
                label=label
            )
        )
    return features

def read_eva_examples(input_file):
    with open(input_file, 'r', encoding='utf-8') as fr:
        examples = []
        for step, line in enumerate(fr):
                raw = line.strip().split("   ")
                if len(raw)==8:
                 user_id = raw[0]
                 news_title = raw[3]
                 news = raw[4]
                 label = int(raw[6])
                 examples.append(InputExample(guid=user_id, text_a=news, text_b=news_title, label=label))
        return examples

def _truncate_seq_pair(tokens_a, tokens_b, rank_tokens,max_length):  # tokens_a 是content, tokens_b是 title ,rank_tokens是rank
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)+len(rank_tokens)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(rank_tokens):
            max=tokens_a
            if len(rank_tokens)>len(tokens_a):
                max=rank_tokens
        else:
            max=tokens_b
            if len(rank_tokens)>len(tokens_b):
                max=rank_tokens
        max.pop()


def accuracy(y_pred, y_true):
    y_pred= np.argmax(y_pred, axis=1)
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    return tot_p,tot_r,tot_f1

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--lstm_hidden_size", default=300, type=int,
                        help="")
    parser.add_argument("--lstm_layers", default=2, type=int,
                        help="")
    parser.add_argument("--lstm_dropout", default=0.5, type=float,
                        help="")

    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--report_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--split_num", default=3, type=int,
                        help="text split")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    # pytorch 中的参数可以直接添加
    args.output_dir='output/'
    args.meta_path = None
    args.do_train = True
    args.do_eval = True
    args.do_test = True
    args.train_steps = 50000
    args.per_gpu_train_batch_size = 64
    args.per_gpu_eval_batch_size = 16
    args.eval_steps = 100
    args.max_seq_length = 250
    args.weight_decay = 0
    args.learning_rate = 5e-6
    args.adam_epsilon = 1e-6
    args.gradient_accumulation_steps = 4

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    try:
        os.makedirs(args.output_dir)
    except:
        pass

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=True)
    config = BertConfig.from_pretrained("bert-base-chinese", num_labels=2)
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', config=config)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)  # 32
    if args.do_train:
        start=time.time()
        train_examples = read_examples('data_new/train_content_rank.txt')  # 都是Inputexamles 对象。
        end=time.time()
        use_time=end-start
        with open("output/time.txt", "a") as writer:
            writer.write("train time  %s"%use_time)
        print("in features")
        train_features = convert_examples_to_features(train_examples, tokenizer,  args.max_seq_length)
        print("out features")
        # 返回的是inputFeatures对象组成的list

        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_mask_ids = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_labels = torch.tensor([f.label for f in train_features])
        print(" all_input_ids_size_before", all_input_ids.size())
        all_input_ids = torch.squeeze(all_input_ids, 1)  # [batchsize,sentence_length]
        print(" all_input_ids_size_after", all_input_ids.size())
        all_mask_ids = torch.squeeze(all_mask_ids, 1)
        all_segment_ids = torch.squeeze(all_segment_ids, 1)
        train_data = TensorDataset(all_input_ids, all_mask_ids, all_segment_ids, all_labels)

        train_dataloader = DataLoader(train_data, shuffle=True,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        num_train_optimization_steps = args.train_steps

        # Prepare optimizer

        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.train_steps)

        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))  # 5872
        logger.info("  Batch size = %d", args.train_batch_size)  # 4
        logger.info("  Num steps = %d", num_train_optimization_steps)  #

        best_acc = 0
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
        train_dataloader = cycle(train_dataloader)

        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask,user_ids, label_ids = batch
            outputs = model(input_ids=input_ids, user_ids=user_ids, attention_mask=input_mask, labels=label_ids)
            loss, logits = outputs[:2]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.fp16 and args.loss_scale != 1.0:
                loss = loss * args.loss_scale
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()  #pytorch中的item 是将零维张量变成浮点数。
            train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)  #返回浮点数四舍五入值，第二个参数表示保留几位
            bar.set_description("loss {} ".format(train_loss))
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1 #梯度累计的step

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()  #反向传播，计算当前梯度

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:

                scheduler.step()
                optimizer.step()      # 根据梯度更新网络参数
                optimizer.zero_grad()  #清空过往梯度
                global_step += 1

            if (step + 1) % (args.eval_steps * args.gradient_accumulation_steps) == 0:
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                logger.info("***** Report result *****")
                logger.info("  %s = %s", 'global_step', str(global_step))
                logger.info("  %s = %s", 'train loss', str(train_loss))

            if args.do_eval and (step + 1) % (args.eval_steps * args.gradient_accumulation_steps) == 0:

                    inference_labels = []
                    gold_labels = []
                    inference_logits = []
                    eval_examples = read_eva_examples('data_new/test_content_rank.txt')
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length)
                    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
                    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
                    all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)

                    all_input_ids = torch.squeeze(all_input_ids, 1)  # [batchsize,sentence_length]

                    all_mask_ids = torch.squeeze(all_input_mask, 1)
                    all_segment_ids = torch.squeeze(all_segment_ids, 1)

                    eval_data = TensorDataset(all_input_ids, all_mask_ids , all_segment_ids, all_label)

                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    eval_dataloader = DataLoader(eval_data, shuffle=True, batch_size=args.eval_batch_size)

                    model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        user_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)

                        with torch.no_grad():
                            outputs = model(input_ids=input_ids, user_ids=user_ids, attention_mask=input_mask,
                                            labels=label_ids)
                            tmp_eval_loss, logits = outputs[:2]
                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        inference_labels.append(np.argmax(logits, axis=1))
                        gold_labels.append(label_ids)
                        inference_logits.append(logits)
                        eval_loss += tmp_eval_loss.mean().item()
                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                    gold_labels = np.concatenate(gold_labels, 0)
                    inference_logits = np.concatenate(inference_logits, 0)
                    model.train()
                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy,eval_recall,eva_F = accuracy(inference_logits, gold_labels)

                    result = {'eval_loss': eval_loss,
                              'eval_accuracy': eval_accuracy,
                              'eval_recall':eval_recall,
                              'eva_F':eva_F,
                              'global_step': global_step,
                              'loss': train_loss}

                    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))
                        writer.write('*' * 80)
                        writer.write('\n')
                    if eva_F > best_acc:
                        print("=" * 80)
                        print("Best F1", eva_F)
                        print("Saving Model......")
                        best_acc = eva_F
                        # Save a trained model
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print("=" * 80)
                    else:
                        print("=" * 80)

if __name__ == "__main__":
    main()
