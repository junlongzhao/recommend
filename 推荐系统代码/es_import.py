from multiprocessing import Process, Queue
from tqdm.auto import tqdm
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import json
import time


consumer_n = 20
buck_size = 40
wiki = "data_json/user_table.json"
totoal_num = 180000  # 60283085#30302#60283085


def import_es(es, lines):
    pages = []
    for line in lines:
        page = json.loads(line)
        page['_index'] = "user"
        page['_type'] = "external"
        pages.append(page)
    res = helpers.bulk(es, pages)

pbar = tqdm(total=totoal_num)

def procducer(q, wikifile, pbar):
    cnt = 0
    print("processing...")
    with open(wikifile,'r',encoding='utf-8') as fr:
     for line in fr:
        pbar.update(max(cnt - q.qsize() - pbar.n, 0))
        q.put(line)
        cnt += 1
        pbar.set_description("Queue Remain: %s" % q.qsize())
    print("total pages:", cnt)

    for i in range(consumer_n):
        q.put(None)
    print("End procducer")


def consumer(q):

    es = Elasticsearch(
        [{'host': '192.168.1.106', 'port': 4445}],
        timeout=50, max_retries=10, retry_on_timeout=True)
    pages = []
    while True:
        res = q.get()
        if res == None or len(pages) >= buck_size:
            import_es(es, pages)
            pages = []
        if res is None:
            break
        pages.append(res)

    print("End consumer")


if __name__ == '__main__':
    q = Queue()
    p = Process(target=procducer, args=(q, wiki, pbar)) #创建生产者进程
    p.start()

    cs = []
    for _ in range(consumer_n):
        c = Process(target=consumer, args=(q,))
        c.start()
        cs.append(c)


    p.join()  #用join方法保证生产者生产完毕

    for c in cs:
        c.join()
    pbar.close()

    print('Finish Processing.')
