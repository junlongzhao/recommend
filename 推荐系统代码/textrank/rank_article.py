from jieba import analyse
from  pyhanlp import *
textrank=analyse.textrank
def rank_article():
    stop_word=[]
    with open('../vocab/Chinese_Stop.txt','r',encoding='utf-8') as fr:
        for word in  fr:
            stop_word.append(word.strip())
    with open('../data_new/textrank_raw.txt','r',encoding='utf-8') as fr:  #加两个点，走到上级目录
        content_list=[]
        for num,line in enumerate(fr):
             content=""
             keywords = textrank(line,topK=20)
             for keyword in keywords:
                 if keyword not  in stop_word:
                    content=content+" "+keyword
             content_list.append(content)
        add = '\n'.join([data.strip() for data in content_list])
        with open("textrank.txt",'w',encoding='utf-8') as fr :
                 fr.write(add)

def rank_article_test():
    test_article=set()
    join_list=[]
    with open('../data_new/train_all_id.txt','r',encoding='utf-8') as fr:
        for num,line in enumerate(fr):
           sentence=line.strip().split("   ")[4]
           test_article.add(sentence)
    for data in test_article:
        join_list.append(data)
        add = '\n'.join([data.strip() for data in join_list])
        with open("../data_new/textrank.txt",'w',encoding='utf-8') as fr :
             fr.write(add)
    with open("../data_new/textrank_raw.txt", 'r', encoding='utf-8') as fr:
       content_list=[]
       for line in fr:
         content=""
         hanlp=HanLP.extractKeyword(line, 20)
         for key in hanlp:
             content=content+"  "+key
         content_list.append(content)
    print("content_list",content_list)
    # with open('../vocab/Chinese_Stop.txt','r',encoding='utf-8') as fr:
    #     stop_words=[]
    #     for line in fr:
    #          stop_words.append(line.strip())
    #
    # with open("../data_new/textrank_wordII.txt", 'w', encoding='utf-8') as fr:
    #   join_list=[]
    #   for data in content_list:
    #       if data not in stop_words:
    #         join_list.append(data)
    #   add = '\n'.join([data.strip() for data in join_list])
    #   fr.write(add)
    #

def update_title_to_summary():
    stop_word = set()
    with open('../vocab/Chinese_Stop.txt', 'r', encoding='utf-8') as fr:
        for word in fr:
            stop_word.add(word.strip())
    with open('../data_new/test_all_id_new.txt', 'r', encoding='utf-8') as fr:
        update_list=[]
        for line in fr:
            content=""
            data = line.strip().split("   ")
            new_content = data[4]
            hanlp = HanLP.extractKeyword(new_content, 20)
            for key in hanlp:
              if key not in stop_word:
               content = content + "  " + key
            data[4]=content
            content_line = data[0] + "   " + data[1] + "   " + data[2] + "   " + data[3] + "   " + \
                  data[4] + "   " + data[5] + "   " + data[6]
            update_list.append(content_line)
        with open('../data_new/test_all_title_rank.txt', 'w', encoding='utf-8') as fr:
         add = '\n'.join([data.strip() for data in update_list])
         fr.write(add)


if __name__=="__main__":
     #rank_article_test()
     update_title_to_summary()
     #rank_article()