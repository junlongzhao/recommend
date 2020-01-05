from my_pytorch_transformers.tokenization_bert import BertTokenizer
from my_pytorch_transformers.modeling_bert import BertForSequenceClassification, BertConfig
from run_bert import convert_examples_to_features
from run_bert import InputExample
from run_bert import select_field
from torch.utils.data import (DataLoader, TensorDataset)
import torch
import argparse
import os
from data_utils import predict2json,keywordpredictjson,groupsort
from data_utils import getmax

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

args = parser.parse_args()
args.max_seq_length = 150
args.output_dir='output/'
device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=True)
config = BertConfig.from_pretrained("bert-base-chinese", num_labels=2)
model = BertForSequenceClassification.from_pretrained(os.path.join(args.output_dir, "pytorch_model.bin"), config=config)
model.to(device)

def news_predict(es,publishDate,userID, count=20, recentDays=10):
        # nowDays = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # publishDate= int(nowDays[0:8])  # 提取年月日
        #publish=["200006051542","200804142218","200301200000"]
        #publishDate=20140330
        examples_content = []; examples_article_id=[];examples_article_title=[]
        for time in range(recentDays): #提取最近10天
            articles = es.search(index='article', body={"query": {"match": {"publishDate": publishDate}}})['hits']
            publishDate = publishDate - 1
            articles=articles['hits']
            for article in articles:
             article=article['_source']
             article_id=article['articleID']
             article_title=article['title']
             news_content=article['news_content']  #取出了最近几天的数据
             examples_article_id.append(article_id)
             examples_content.append(news_content)
             examples_article_title.append(article_title)
        #print("examples_article_id",examples_article_id)
        print("length examples_article_id",len(examples_article_id))

        click_foward_predict(userID,examples_article_title,examples_content, examples_article_id,count)

def foward(userID,examples_article_title,examples_content,article_id,count, id_all_list_mapping=None, news_publishtime_cluster_all_list=None):
            examples=[];num=0; examples_sort=[]
            for news_content,title in zip(examples_content,examples_article_title):
              examples.append(InputExample(guid=userID, text_a=news_content, text_b=title))
            predict_features = convert_examples_to_features(examples, tokenizer, args.max_seq_length)
            all_input_ids = torch.tensor(select_field(predict_features, 'input_ids'), dtype=torch.long)
            all_input_mask = torch.tensor(select_field(predict_features, 'input_mask'), dtype=torch.long)
            all_segment_ids = torch.tensor(select_field(predict_features, 'segment_ids'), dtype=torch.long)
            all_input_ids = torch.squeeze(all_input_ids, 1)  # [batchsize,sentence_length]
            all_mask_ids = torch.squeeze(all_input_mask, 1)
            all_user_ids = torch.squeeze(all_segment_ids, 1) #userid
            predict_data=TensorDataset(all_input_ids,all_mask_ids,all_user_ids)
            predict_dataloader = DataLoader(predict_data, shuffle=False)
            for input_ids, input_mask, user_ids in  predict_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                user_ids= user_ids.to(device)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, user_ids=user_ids, attention_mask=input_mask)
                    logits = outputs[:1]
                    logits=logits[0]
                    logits = logits.cpu().numpy()
                    predictValue=getmax(logits[0])*100 #predictValue
                    if id_all_list_mapping!=None: #聚类簇的情况
                      article_predict=(article_id[num],predictValue,news_publishtime_cluster_all_list[num]) #做成tuple,然后降序排列
                      num=num+1
                      examples_sort.append(tuple(article_predict))
                    if id_all_list_mapping==None: #点击预测的情况
                        article_predict = (article_id[num], predictValue)
                        num=num+1
                        examples_sort.append(tuple(article_predict))
            examples_sort.sort(key = lambda examples_sort: examples_sort[1],reverse=True)
            json_article=[];json_article_id=[];json_predict=[];json_time=[]
            for step,article_predict in enumerate(examples_sort): #从元组中取出id和评分,这是排序之后的结果,这是所有的
               if step<=count:
                article_id,predict,time=article_predict
                article="article"+str(step)
                json_article.append(article)
                json_article_id.append(article_id)
                json_predict.append(predict)
                json_time.append(time)
            if id_all_list_mapping!=None:
              id_cluster_all=[];new_cluster_all=[];time_cluster_all=[]
              for row in range(len(id_all_list_mapping)): #把每一个簇里面的数据取出来
                 news_cluster_predict=[];id_cluster=[];publishTime_cluster=[]
                 for id in id_all_list_mapping[row]:
                     index=json_article_id.index(id) #在所有列表里面找到分组这个中的下标
                     predict=json_predict[index]
                     time=json_time[index]
                     news_cluster_predict.append(predict)
                     id_cluster.append(id)
                     publishTime_cluster.append(time)
                 id_cluster_all.append(id_cluster)
                 new_cluster_all.append(news_cluster_predict)
                 time_cluster_all.append(publishTime_cluster)
              cluster_all = []
              for row in range(len(id_cluster_all)):
                    cluster_sort=[]
                    for num in range(len(id_cluster_all[row])):
                        article_predict = (id_cluster_all[row][num], new_cluster_all[row][num],time_cluster_all[row][num])  # 做成tuple,然后降序排列
                        cluster_sort.append(tuple(article_predict))
                    cluster_sort.sort(key=lambda cluster_sort: cluster_sort[1], reverse=True) #组内兴趣值排列
                    cluster=keywordpredictjson(cluster_sort)
                    cluster_all.append(cluster)
              groupsort(cluster_all)

def click_foward_predict(userID,examples_article_title,examples_content,article_id,count):
    examples = [];num = 0
    examples_sort = []
    for news_content, title in zip(examples_content, examples_article_title):
        examples.append(InputExample(guid=userID, text_a=news_content, text_b=title))
    predict_features = convert_examples_to_features(examples, tokenizer, args.max_seq_length)
    all_input_ids = torch.tensor(select_field(predict_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(predict_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(predict_features, 'segment_ids'), dtype=torch.long)
    all_input_ids = torch.squeeze(all_input_ids, 1)  # [batchsize,sentence_length]
    all_mask_ids = torch.squeeze(all_input_mask, 1)
    all_user_ids = torch.squeeze(all_segment_ids, 1)  # userid
    predict_data = TensorDataset(all_input_ids, all_mask_ids, all_user_ids)
    predict_dataloader = DataLoader(predict_data, shuffle=False)
    for input_ids, input_mask, user_ids in predict_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        user_ids = user_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, user_ids=user_ids, attention_mask=input_mask)
            logits = outputs[:1]
            logits = logits[0]
            logits = logits.cpu().numpy()
            predictValue = getmax(logits[0]) * 100  # predictValue
            article_click_predict = (article_id[num], predictValue)
            num = num + 1
            examples_sort.append(tuple(article_click_predict))
    examples_sort.sort(key=lambda examples_sort: examples_sort[1], reverse=True)
    json_article = [];json_article_id = [];json_predict = []
    for step, article_predict in enumerate(examples_sort):  # 从元组中取出id和评分,这是排序之后的结果,这是所有的
             if step <= count:
                 article_id, predict= article_predict
                 article = "article" + str(step)
                 json_article.append(article)
                 json_article_id.append(article_id)
                 json_predict.append(predict)
             predict_list = predict2json(json_article, json_article_id, json_predict)
    print("predict_list",predict_list)

