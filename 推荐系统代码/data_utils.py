import jieba
import  numpy as np
import json
import copy
import codecs
def bulid_date(special_flag=False):
    """  False表示是正常的行，True表示该行不正常，可能时间为空那种
     提取出文本中新闻的发布时间，并且按照时间顺序排序
    :return:
    """
    with open('data/train._clean3.txt',encoding='utf-8') as fr:
        count_num = 0;time_all_news=[];all_news_id=[];special_line=[];
        for num,line in  enumerate(fr):
           content= line.strip().split("	")
           time=content[5] #这里是新闻发布时间
           all_news_id.append(content[1])
           cut_time=jieba.lcut(time)
           if (cut_time[0]=="NULL" or len(cut_time)!=9) and special_flag:
                special_line.append(line)
           if cut_time[0]!="NULL" and len(cut_time)==9:
              count_num = count_num + 1
              fulltime=cut_time[0]+cut_time[2]+cut_time[4]+cut_time[6]+cut_time[8] #年，月，日，时，分
              content[5]=content[5].replace(content[5],fulltime)
              time_all_news.append(content)
        time_all_news.sort(key=lambda time_all_news: time_all_news[5]) #对正常时间序列进行排序
        if special_flag :
            content = '\n'.join([line.strip() for line in special_line])
            with open("data/train_clean.txt",'w',encoding='utf-8') as fr:   #添加时间不规则的数据
              fr.write(content)
        if not special_flag:
            all_regular_line=[]
            for line in time_all_news:
              line=line[0]+"   "+line[1]+"   "+line[2]+"   "+line[3]+"   "+line[4]+"   "+line[5]+line[6].strip()
              all_regular_line.append(line)
            content_regular='\n'.join([line.strip() for line in all_regular_line])
            with open("data/train_clean.txt",'a',encoding='utf-8') as fr:
                   fr.write(content_regular)

def write():
    linelist=[]
    with open('data/train_clean.txt','r',encoding='utf-8') as fr:
        for step,line in enumerate(fr):
            if step<=4000:
             linelist.append(line)
    with open('data/train_three.txt','w',encoding='utf-8') as fr:
            content='\n'.join([line.strip() for line in linelist])
            fr.write(content)

def testusers():  #构造负样本
    content=[];summary=[]
    with open('data/train_clean.txt','r',encoding='utf-8') as fr:
        for  line in fr:
            split_data=line.strip().split("   ")
            if line is not None or line != "":
             content.append(split_data)
    with open('data/train_clean2.txt','r',encoding='utf-8') as fr1:
        for line in fr1:
            split_data_2=line.strip().split("   ")
            if line is not None or line != "":
              summary.append(split_data_2)

    assert len(summary) == len(content)
    max_index=len(content)
    write_false=[]
    numconut=0
    for data1 in content:
        num=np.random.randint(0, max_index, size=5).tolist() #随机取了5个下标
        for num_index in num:
           if summary[num_index][4]!=data1[4] and summary[num_index][0]!=data1[0] and numconut<=100000:  # content 和id都不一样
               numconut = numconut + 1
               data1[0]=summary[num_index][0]
               content_all = data1[0] + "   " + data1[1] + "   " + data1[2] + "   " + data1[3] + "   " + \
                             data1[4] + "   " + data1[5]
               write_false.append(content_all)
    false_id_content="\n".join([data.strip() for data in write_false])
    print(false_id_content)
    with open("train_false.txt",'w',encoding='utf-8') as fr:
         fr.write(false_id_content)

def addsample():  #将正负样本合并在一起
     samplelist=[]
     with open('train_false.txt','r',encoding='utf-8') as fr:
         for line in fr:
              line=line.strip()+"   "+"0"
              samplelist.append(line)
     with open('data/train_clean3.txt','a',encoding='utf-8')   as fr:
              add='\n'.join([data.strip() for data in samplelist])
              fr.write(add)

def timesort():
    time_all_news=[]
    with open('data/train_clean3.txt','r', encoding='utf-8') as fr:
        for line in fr:
           data=line.strip().split("   ")
           content=data[0]+"   "+data[1]+"   "+data[2]+"   "+data[3]+"   "+data[4]+"   "+data[5]+"   "+data[6]
           time_all_news.append(content)
    time_all_news.sort(key=lambda time_all_news: time_all_news[5])
    with open('data/train_final.txt','w',encoding='utf-8') as fr:
          write_all='\n'.join([data.strip() for data in time_all_news ])
          fr.write(write_all)

def text2json():#把数据做成json格式,即每一行都是一个字典
    data_list=[]
    with open('data_new/train_content_rank.txt','r',encoding='utf-8') as fr:
        for step,line in enumerate(fr):
            data = {}
            raw=line.strip().split("   ")
            if len(raw)==8:    #总共有8个字段
             data['user_id'] =raw[0]
             data['news_id']=raw[1]
             data['browsing_time']=raw[2]
             data['news_title']=raw[3]
             data['news_content']=raw[4]
             data['news_publish']=raw[5]
             data['news_label']=raw[6]
             data['news_rank_text']=raw[7]
             data_list.append(data)
    with codecs.open('data.json','a','utf-8') as json_file:
        for each_dict in data_list:
            json_file.write(json.dumps(each_dict,ensure_ascii=False)+"\n")

def jsontest():
    wordid=[]
    word2id={}
    word=['zhangsan','lisi','wangmazi','zhaoliu']
    id=['1','2','3','4']
    for word,id in zip(word,id):
        word2id[word]=id
        wordid.append(word2id)
    with open('word2id.json','w') as fr:
        json.dump(word2id,fr)

def dataToJson(title,content, publishDate,articleID):
    """
    :param title:
    :param content:
    :param publishDate:
    :return:json
    """
    JsonNews={}
    JsonNews['articleID'] = articleID
    JsonNews['title']=title
    JsonNews['news_content']=content
    JsonNews['publishDate']=publishDate

    return JsonNews

def UserdataToJson(userID, articleID, clickDate):
    JsonUser={}

    JsonUser['articleID'] = articleID
    JsonUser['clickDate'] = clickDate
    JsonUser['UserID'] = userID
    return JsonUser

def getmax(logits):
    if logits[0]>logits[1]:
        return logits[0]
    else :
        return logits[1]

def predict2json(article,article_id,predict):
     predict_list=[]
     step=0
     for artilce,id,predict in zip(article,article_id,predict):
         json = {}
         json['article']=step
         json["articleID"]=id
         json["predictValue"]=predict
         step=step+1
         predict_list.append(json)
     return predict_list

def keywordpredictjson(cluster_sort):
     predict_list=[]
     for step,id_predict_time in enumerate(cluster_sort):
         json={}
         id,predict,time=id_predict_time
         json["articleID"]=id
         json["predictValue"]=predict
         json["time"]=time
         predict_list.append(json)
     return predict_list

def groupsort(cluster_all): #组间根据发布日期排序,取最大的时间排在最前面
    max_time_list=[]
    for step,article_list in enumerate(cluster_all): #这是一个分好簇的二维数组
      max_time=0
      for article in article_list:
          time=int(article['time'])
          #print("time", time)
          if time>max_time: #得到每一组内最大的time
              max_time=time
      max_time_list.append(max_time)
    new = max_time_list.copy()
    list.sort(new,reverse=True) #浅拷贝
    cluster_list=[]
    for step,number in enumerate(new):
        cluster_dict = {}
        cluster="cluster"+str(step)
        index=max_time_list.index(number)
        cluster_dict[cluster] = cluster_all[index]
        cluster_list.append(cluster_dict)
    print("cluster",cluster_list)


if __name__=="__main__":
    #bulid_date(False)
     #write()
     testusers()
    # addsample()
     #jsontest()
     #text2json()