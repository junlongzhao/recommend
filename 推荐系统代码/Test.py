import re
import jieba
import  numpy as np
from my_pytorch_transformers import AutoConfig,AutoModel
import torch
import datetime
import time
from time import sleep
import os
from multiprocessing import Pool,Process,Queue


def TestNewsid(all_news_id): #新闻有重复，一共有6183条新闻
    test=set()
    for news_id in all_news_id:
        test.add(news_id)
    print(len(test))

def Testdata():#测试做好的数据能否正常切割
    with open('data/train_sort.txt',encoding='utf-8') as fr:
        for line in fr:
          split_line=line.split("   ")
          print(len(split_line))


def testnull():
    a='123张云'
    a=re.sub("\D", "", a) #除去非数字
    print(int(a))

def testjieba():  #清除中间空格的方式。
    a="我是一个来自北方的狼       ，在这里被         冷成了狗"
    a=a.replace(" ","")
    data=jieba.lcut(a.strip())
    print(data)


def testusers():
    content=[];summary=[]
    with open('data/train_clean.txt','r',encoding='utf-8') as fr:
        for  line in fr:
            split_data=line.strip().split("   ")
            if line is not None or line != "":
             content.append(split_data)
    # for data in content:
    #     print(data[4])
    with open('data/train_clean2.txt','r',encoding='utf-8') as fr1:
        for line in fr1:
            split_data_2=line.strip().split("   ")
            if line is not None or line != "":
              summary.append(split_data_2)

    assert len(summary) == len(content)
    max_index=len(content)
    write_false=[];write_false_list=[]
    numconut=0
    for data1 in content:
        #print("data1 before",data1)
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

def testset():
    a=['1','2','3','4','5','6']
    testset=set()
    for i in a:
        testset.add(i)
    for j in testset:
      for k in testset:
         if j!=k:
             print("j=%2s k=%2s"%(j,k))

def testlength():
    with open('data/train_clean.txt', 'r', encoding='utf-8') as fr:
        usernum=set()
        for line in fr:
           data=line.strip().split("   ")
           usernum.add(data[4])
        print(len(usernum))

def testsample():
    neg_count=0
    pos_count=0
    with open('data/train_final.txt','r',encoding='utf-8') as fr:
        for step,line in enumerate(fr):
           if step>=180000:    #总体的数据量在20万，这里写死了，取后面的百分之十
             data= int (line.strip().split("   ")[6])
             #print("time%2s label%2s"%(data[5],data[6]))
             if data==1:
                pos_count+=1
             else:
                 neg_count+=1
    print("pos_count", pos_count)
    print("neg_count",neg_count)

def autoconfig():
   config=AutoConfig.from_pretrained('model/chinese_L-12_H-768_A-12/bert_config.json')
   model=AutoModel.from_pretrained('./model/chinese_L-12_H-768_A-12/bert_model',from_tf=True, config=config)
   print(model)

def testdata(): #将id转换成二进制
    data_list=[]
    with open('data/train_final.txt','r',encoding='utf-8') as fr:
         for step,line in enumerate(fr):
           if step<=180000:
            data=line.strip().split("   ")
            if len(data)==7:
             # data[0] =re.sub("\D", "", data[0])
             # data[0]=bin(data[0])
             data_write=data[0] + "   " + data[1] + "   " + data[2] + "   " + data[3] + "   " + \
                             data[4] + "   " + data[5]+"   "+data[6]
             data_list.append(data_write)
    data_bin = "\n".join([data.strip() for data in  data_list])

    with open("train_.txt", 'w', encoding='utf-8') as fr:
        fr.write(data_bin)

def bin(n):
    x = 2  # 转换为二进制，所以这里取x=2
    b = []  # 存储余数
    n=int(n)
    while True:  # 一直循环，商为0时利用break退出循环
        s = n // x  # 商
        y = n % x  # 余数
        b = b + [y]  # 每一个余数存储到b
        if s == 0:
            break  # 余数为0时结束循环
        n = s
    b.reverse()  # 使b中的元素反向排列
    result=""
    for i in b:
        result=result+str(i)
    return  result

def AddDim():
    ones=torch.ones([4,5,3],dtype=torch.long)
    zeros=torch.zeros([4,1,3],dtype=torch.long)
    catresult=torch.cat((ones,zeros),dim=1)
    print(ones)
    print(zeros)
    print(catresult)

def thread():
   # a=torch.rand(1,27).reshape([3,3,3])
   # b=a[:,0,:]
   # print(a)
   # print(b)

   a=torch.ones((3,3))
   b=torch.zeros((3,3))
   c=torch.cat((a,b),dim=-1)
   print(c)

def data():
    with open('data_new/test_all.txt','r',encoding='utf-8') as fr :
        user_dataset=set()
        user_list=[]
        for line in fr:
           try:
               if line == "" or line is None:
                   continue
               data=line.strip().split("   ")
               user_list.append(data[0])
               user_dataset.add(data[0])
           except Exception:
             print("出现错误", line)
             continue
        print("user number",len(user_dataset))
        user_index=0
    user_dataset_list=list(user_dataset)
    #print("user_dataset_list_length",len(user_dataset_list))#8625
    with open('data_new/test_all.txt', 'r', encoding='utf-8') as fr:
      data_list = [];count_zero=0;count_one=0;
      for num,line in enumerate(fr):
        if num<=1000:
         data=line.strip().split("   ")
         assert user_list[user_index]==data[0]
         user_index += 1
         data[0]=str(user_dataset_list.index(data[0]))
         data_all=data[0] + "   " + data[1] + "   " + data[2] + "   " + data[3] + "   " + \
                             data[4] + "   " + data[5]+"   "+data[6]
         if int(data[6])==0:
              count_zero=count_zero+1
         if int(data[6])==1:
             count_one=count_one+1
    print("count_one",count_one)
    print("count_zero",count_zero)
    #      data_list.append(data_all.strip())
    # data_userid = "\n".join([data.strip() for data in data_list])
    # with open('data_new/test_all_id.txt', 'w', encoding='utf-8') as fr:
    #         fr.write(data_userid)

def tupletest():
    with open('data_new/test_all_id.txt', 'r', encoding='utf-8') as fr:
        for num,line in enumerate(fr):
            date = line.strip().split("   ")[5]
            print(date)

def testid():
    zerocount=0;onecount=0
    with open('data_new/train_all_id.txt', 'r', encoding='utf-8') as fr:
        for num,line in enumerate(fr):
           if num<=50000:
            data = line.strip().split("   ")
            if int(data[6])!=0 and int(data[6])!=1: #这种判断要做好，不然可能会出现越界的情况
                print("num%d line%s"%(num,line))
            if int(data[6]) ==0:
                zerocount+=1
            if  int(data[6])==1:
                onecount+=1
        print("zerocount",zerocount)
        print("onecount",onecount)

def updatedata():
    data_list=[]
    with open('data_new/train_all_id.txt', 'r', encoding='utf-8') as fr:
        for num,line in enumerate(fr):
            data=line.strip().split("   ")
            content=data[0] + "   " + data[1] + "   " + data[2] + "   " + data[3] + "   " + \
                              data[4] + "   " + data[5]+"   "+data[6]
            if int(data[6])!=0 and int(data[6])!=1:
                print("num",num)
                continue
            else:
               data_list.append(content)
    print(len(data_list))

    # with open('data_new/train_all_id.txt','w',encoding='utf-8') as fr:
    #          write_all="\n".join([data for data in data_list])
    #          fr.write(write_all)

def nulltest():
    with open('data_new/train_all_id_new.txt', 'r', encoding='utf-8') as fr:
        for line in fr:
            data = line.strip().split("   ")
            news = data[4]
            if news == "NULL":
               print(line)

def test_user_number():
    uernum=set()
    with open('data_new/test_all_id.txt', 'r', encoding='utf-8') as fr:
             for line in fr:
                 news_title=line.strip().split("   ")[3]
                 uernum.add(news_title)
             print(len(uernum))


def deletenull():
    all_content_list=[]
    with open('data_new/test_all_id.txt','r',encoding='utf-8') as fr:
        for line in fr:
            data=line.strip().split("   ")
            news=data[4]
            if news!="NULL" :
                content = data[0] + "   " + data[1] + "   " + data[2] + "   " + data[3] + "   " + \
                          data[4] + "   " + data[5] + "   " + data[6]
                all_content_list.append(content)
        add='\n'.join(data for data in all_content_list)
    with open('data_new/test_all_id_new.txt','w',encoding='utf-8') as fr:
        fr.write(add)

def rank_content():
    all_content_list=[];all_content_list_rank=[];all_content_list_addrank=[]
    with open('data_new/test_all_id_new.txt','r',encoding='utf-8') as fr:
        for line in fr:
            data = line.strip().split("   ")
            content = data[0] + "   " + data[1] + "   " + data[2] + "   " + data[3] + "   " + \
                      data[4] + "   " + data[5] + "   " + data[6]
            all_content_list.append(content)
    with open('data_new/test_all_title_rank.txt', 'r', encoding='utf-8') as fr:
         for line in fr:
             raw = line.strip().split("   ")
             content_rank = raw[0] + "   " + raw[1] + "   " + raw[2] + "   " + raw[3] + "   " + \
                       raw[4] + "   " + raw[5] + "   " + raw[6]
             all_content_list_rank.append(content_rank)
    for num in range(len(all_content_list_rank)):
         raw=all_content_list[num].strip().split("   ")
         rank=all_content_list_rank[num].strip().split("   ")
         assert raw[0]==rank[0]
         content_rank = raw[0] + "   " + raw[1] + "   " + raw[2] + "   " + raw[3] + "   " + \
                        raw[4] + "   " + raw[5] + "   " + raw[6]+ "   "+rank[4]
         all_content_list_addrank.append(content_rank)
    add="\n".join([data.strip() for data in all_content_list_addrank])
    with open("data_new/test_content_rank.txt",'w',encoding='utf-8') as fr:
        fr.write(add)

def rank_title():
    with open("data_new/train_content_rank.txt",'r',encoding='utf-8') as fr:
        for line in fr:
           raw=line.strip().split("   ")
           print("length",len(raw))


def delete_max_length():
    a=np.arange(1,30).tolist()
    b=np.arange(3,34).tolist()
    c=np.arange(9,27).tolist()
    if len(a)>len(b):
        max=a
        if len(c)>len(a):
            max=c
    else:
         max=b
         if len(c)>len(b):
             max=c
    max.pop()
    print(max)
    print("a",a)
    print("b",b)
    print("c",c)


def random():
    nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def recnentDays():
    nowdays=datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    nowdays=int(nowdays[0:8]) #提取年月日
    print("nowdays length",nowdays)
    for day in range(10):
         print(nowdays)
         nowdays=nowdays-1


def sort_test():
    a=[(1,'a'),(2,'b')]
    for x,y in a:
        print(x)
        print(y)

def write(q):
    print("启动Write子进程：%s" % os.getpid())
    for i in ["A", "B", "C", "D"]:
        q.put(i)  # 写入队列
        time.sleep(1)
    print("结束Write子进程：%s" % os.getpid())

def read(q):
    print("启动read子进程：%s" % os.getpid())
    while True:  # 阻塞，等待获取write的值
        value = q.get()
        print(value)
    print("结束read子进程：%s" % os.getpid())  # 不会执行


def test_list_length():
    a=[[1,2,3],[4,5,6],[7,8,9]]
    print(a[1][1])


def copy():
    old = [1, [1, 2, 3], 3]
    new = old.copy()
    print('Before:')
    print(old)
    print(new)
    new[0] = 3
    new[1][0] = 3
    print('After:')
    print(old)
    print(new)


#After:
#[1, [3, 2, 3], 3]    外层浅拷贝，内层深拷贝
#[3, [3, 2, 3], 3]

# if __name__ == "__main__":
#     q=Queue()
#     pw=Process(target=write,args=(q,))
#     pr = Process(target=read, args=(q,))
#     pw.start()
#     pr.start()
#     pw.join()
#     pr.terminate() #强行终于读进程
#     print("父进程结束")


if __name__=="__main__":
   copy()