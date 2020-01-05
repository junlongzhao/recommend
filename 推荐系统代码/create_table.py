import json
import codecs
def create_article_table():
    id_list=[];article_set=set();article_list=[]
    with open('data_new/train_content_rank.txt','r',encoding='utf-8') as fr:
           for line in fr:
               article = {}
               raw=line.strip().split("   ")
               if len(raw)==8 and raw[1] not in id_list:
                print("id",raw[1])
                id_list.append(raw[1])
                article['articleID'] = raw[1]  # 文章id
                article['title']=raw[3]    #文章title
                article['news_content']=raw[4]  #文章content
                time=raw[5]
                article['publishDate']=time[0:8]  #文章publishDate
                article['textrank']=raw[7]
                article_list.append(article)
    with codecs.open('data_json/article_table.json','a','utf-8') as json_file:
        for each_dict in article_list:
            json_file.write(json.dumps(each_dict,ensure_ascii=False)+"\n")

def create_user_table():
     user_data_list=[]
     with open('data_new/train_content_rank.txt', 'r', encoding='utf-8') as fr:
         for line in fr:
             user={}
             raw=line.strip().split("   ")
             if len(raw)==8:
              user['articleID']=raw[1]
              user['clickDate']=raw[2]
              user['UserID'] =raw[0]
              user_data_list.append(user)

     with codecs.open('data_json/user_table.json', 'a', 'utf-8') as json_file:
         for each_dict in user_data_list:
             json_file.write(json.dumps(each_dict, ensure_ascii=False) + "\n")


if __name__=="__main__":
    create_user_table()