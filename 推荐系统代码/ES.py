# -*- coding:utf-8 -*-

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from data_utils import dataToJson,UserdataToJson
import datetime
import time
import json
from kmeans.KmeansClustering import KmeansClustering
from predict import news_predict,foward
from run_bert import main
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

class ES():
   def __init__(self, hosts='192.168.1.106:4445'):
           self.hosts = hosts
           self.es = Elasticsearch(hosts=hosts,timeout=20)
           self.search = Search().using(self.es)

   def create_index(self,index_name):
     self.es.indices.create(index=index_name, ignore=400)  # 创建索引节点

   def data_json_insert(self):
     with open('data_json/article_table.json','r',encoding='utf-8') as fr:
      for step,line in enumerate(fr):
        data=json.loads(line) #将str转换成dict
        self.es.index(index='article', doc_type='external', body=data)

   def data_json_search(self,publishDate=None): #查找数据
    """
    :param news_id:新闻的id
    :return:
    """

    print("publishDate",publishDate)
    result = self.es.search(index='article', body={"query": {"match": {"publishDate":publishDate}}}) #查找具体的数据
    print("result ",result)

   def updateTest(self,articleID,news):
       self.es.update(index='recommend',id=articleID,doc_type='external',body=news)


   def delete_index(self,index_name): #删除索引节点
     """
     :param:index_name 索引节点的名称
     :return:
     """
     self.es.indices.delete(index=index_name, ignore=[400, 404])

   def key_word_search(self):
       news_list=[]
       search_task=Search(using=self.es,index='article',doc_type="external").query("match",news_content="李克强")
       for step,hit  in enumerate(search_task):
           print("news_content",hit.news_content)
           news_list.append(hit.news_content)
       print("news_list_length",len((news_list)))

   def uploadArticle(self,title=None, content=None, publishDate=None):  #上传新闻文档

       """
       :param title:
       :param content:
       :param publishDate:
       :return: articleID
       """
       articleID={}
       ID = datetime.datetime.now().strftime("%Y%m%d%H%M%S") #根据时间生成的随机数作为ArticleID
       print("ID",ID)
       news=dataToJson(title, content, publishDate,ID)
       self.es.index(index='article', doc_type='external',body=news)
       articleID["articleID"]=ID
       print(articleID)
       return articleID

   def deleteArticle(self,articleID): #删除具体的新闻文档
     """
     :param:news_id 新闻的id
     :return:
     """
     result = self.es.search(index="article", body={"query": {"match": {"articleID": articleID}}})['hits']
     result=result['hits']
     result = result[0]
     id = result['_id'] #先查询获取id
     self.es.delete(index='article',doc_type='external',id=id)

   def updateArticle(self,title, content, publishDate, articleID):  #对文档做更新
       """

       :param title:
       :param content:
       :param publishDate:
       :param articleID:
       :return: null
       """
       news=dataToJson(title, content, publishDate,articleID)
       result = self.es.search(index="article", body={"query": {"match": {"articleID": articleID}}})['hits']
       result = result['hits']
       result = result[0]
       id = result['_id']  # 先查询获取id
       self.es.delete(index='article',doc_type='external',id=id)

       #self.es.index(index="article",id=articleID,doc_type='external',body={"doc":news})
       time.sleep(1)
       self.es.index(index = 'article', doc_type = 'external', body = news)

   def queryArticle(self,ArticleId): #查询数据
       """

       :param ArticleId:
       :return: {""title":"  " ,"content": " "  ,"publishDate":"  " ,"articleID":" " }
       """
       result = self.es.search(index="article", body={"query": {"match": {"articleID": ArticleId}}})['hits']
       print(result)
       result=result['hits']
       if result!=[]:
         result=result[0]
         result=result['_source']
       else:
           print("未查询到数据")
       return result

   def UserClick(self,articleID,clickDate,userID): #上传用户点击数据
       """
       :param userID:
       :param articleID:
       :param clickDate:
       :return:
       """
       user=UserdataToJson(userID, articleID, clickDate)
       self.es.index(index='user',doc_type='external', body=user)


   def getUserClickHistory(self,userID):
       """
       :param userID:
       :return:[{"userID,"articleId","clickDate"}]
       """
       all_user=[]
       result = self.es.search(index="user", body={"query": {"match": { "UserID": userID}}})['hits']
       result = result['hits']   #此处会形成一个list，关于user
       for user in result:
          user= user['_source']
          all_user.append(user)
       print("all_user",all_user)
       return  all_user


   def trainModel(self): #重新训练模型
       main()


   def  clickPredict(self,publishDate,userID, count=20,recentDays=10): #点击预测，返回recentDays的count个articleID。
       news_predict(self.es,publishDate,userID, count, recentDays)


   def  searchArticlepredict(self,userID,searchWord,count=20,clusterCount=5, sortBy='Interset', clusterSortBy='Date'):
       search_task = Search(using=self.es, index='article', doc_type="external").query("match", news_content=searchWord).extra(size=count)
       news_content_list=[]; news_title_list=[];news_id_list=[];news_date_list=[]
       for step, hit in enumerate(search_task):
           news_content_list.append(hit.news_content)
           news_title_list.append(hit.title)
           news_id_list.append(hit.articleID)
           news_date_list.append(hit.publishDate)
       print("news_data_list",news_date_list)
       Kmeans = KmeansClustering(stopwords_path='data_new/stop_words.txt')
       result = Kmeans.kmeans(news_content_list, n_clusters=clusterCount) #聚合结果根据读取顺序，重新做成了0到count的序列
       id_all_list_mapping=[];id_cluster_all_list=[]; news_title_cluster_all_list=[];news_content_cluster_all_list=[]
       news_publishtime_cluster_all_list=[]
       if sortBy=='Interset' and clusterSortBy=='Date':
        for key,value in result.items():  #由于聚合的原因导致序列混乱，所以需要重新排列,此处是根据兴趣值
           id_one_list_mapping = []
           for id in value:
                 id_one_list_mapping.append(news_id_list[id])
                 id_cluster_all_list.append(news_id_list[id])
                 news_title_cluster_all_list.append(news_title_list[id])
                 news_content_cluster_all_list.append(news_content_list[id])
                 news_publishtime_cluster_all_list.append(news_date_list[id])
           id_all_list_mapping.append(id_one_list_mapping) #此处是为了后面聚类团做映射

        foward(userID,news_title_cluster_all_list,news_content_cluster_all_list,id_cluster_all_list,count, id_all_list_mapping, news_publishtime_cluster_all_list)



if __name__=="__main__":

    es=ES()
    #es.UserClick('123123','20201011','00')
    #es.queryArticle(20200102160628)
    #es.uploadArticle('测试','对es做测试',20010212)
    #es.updateArticle('测试','对es做测试,对它更新',20010212,'20200102160628')
    #es.deleteArticle(20200102160628)
    #es.getUserClickHistory('480')
    #es.key_word_search() 关键词搜索
    #es.create_index('article')
    #es.data_json_insert()
    #es.clickPredict(20140219,'480')
    #es.searchArticlepredict('480','李克强')
   # es.uploadArticle('中国','中国拥有悠久的历史','20180213')
    #es.updateArticle('中国','中国拥有悠久的历史,但是.。。。','20180213','20191226234825')

