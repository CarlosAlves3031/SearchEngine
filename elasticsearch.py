
import time
import re
import timeit
import matplotlib.pyplot as plt
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from urllib3 import disable_warnings, exceptions


class Text:
    def __init__(self, original):
        self.index, self.title, self.author, self.bibliography, self.body, *_ = re.split(r'.T|.A|.B|.W', original.replace('\n', ' '))

class Query:
    def __init__(self, text):
        self.index, self.body = map(lambda x: x.strip().replace('\n', ' '), text.split('\n.W\n'))

def parse_file(filename, class_type):
    items = []
    with open(filename, 'r') as file:
        txt = file.read()
        txt = txt.split('.I')[1:]
        items = list(map(lambda x: class_type(x), txt))
    return items

def get_ordered_relevant_searches(filename):
    query_relations = {}
    with open(filename, 'r') as file:
        txt = file.read().strip().split('\n')
        for i in txt:
            query, abstract, score = map(int, filter(lambda x: len(x) > 0, i.strip().split(' ')))
            query_relations.setdefault(query - 1, []).append((abstract, score))

    for i in query_relations:
        query_relations[i].sort(key=lambda x: x[1])

    return query_relations

def create_index(es, index_name, mapping):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body={"mappings": mapping})

def add_documents_to_index(es, index_name, documents):
    actions = [
        {
            "_index": index_name,
            "_id": word.index,
            "title": word.title,
            "author": word.author,
            "bibliography": word.bibliography,
            "body": word.body
        }
        for word in documents
    ]

    success, failed = bulk(es, actions=actions)
    return not failed

def search_results(es, index_name, queries, limits):
    results_dict = {}
    for i, (limit, query_to_parse) in enumerate(zip(limits, queries)):
        query = {
            "query": {
                "multi_match": {
                    "query": query_to_parse.body,
                    "fields": ["title", "author", "body"]
                }
            }
        }
        results = es.search(index=index_name, body=query, size=max(limit, 10))
        hits = results.get("hits", {}).get("hits", [])
        results_dict[i] = [(int(hit["_id"]), hit["_score"]) for hit in hits]
    return results_dict

def precision_recall_at_k(answer, relevant, k=None):
    if k is None or k > len(answer) or k > len(relevant):
        k = min(len(answer), len(relevant))

    common_elements = set(answer[:k]) & set(relevant)
    precision = len(common_elements) / k if k != 0 else 0
    recall = len(common_elements) / len(relevant) if len(relevant) != 0 else 0

    return precision, recall

def all_results_by_func(answer_results_dict, relevant_results_dict, func, k=None):
    size = len(relevant_results_dict)
    results = np.zeros((size, 2))
    for i, (answer, relevant) in enumerate(zip(answer_results_dict.values(), relevant_results_dict.values())):
        answer = list(map(lambda x: x[0], answer))
        relevant = list(map(lambda x: x[0], relevant))
        precision, recall = func(answer, relevant, k)
        results[i, 0] = precision
        results[i, 1] = recall
    return results

def plot_results(answer_results_dict, relevant_results_dict, func, k_s=range(1, 10 + 1), title=''):
    results = [all_results_by_func(answer_results_dict, relevant_results_dict, func, k).mean() for k in k_s]
    plt.plot(k_s, results, marker='o')
    plt.xlabel('K')
    plt.ylabel(f'{func.__name__} mean')
    plt.title(title)
    plt.show()

INDEX_NAME = "index_dir2"
# Desabilitando os avisos de certificado
disable_warnings(exceptions.InsecureRequestWarning)

# Conectando ao Elasticsearch
es = Elasticsearch(['https://elastic:xbiw*OpQ4=As+LssZtdm@localhost:9200'], verify_certs=False)
# OBTENDO QUERIES, PALAVRAS E BUSCAS RELEVANTES
queries = parse_file('cran/query.txt', Query)
words = parse_file('cran/dataset.txt', Text)
relevant_dict = get_ordered_relevant_searches('cran/gabarito.txt')

# INDEXANDO RESULTADOS
t0 = timeit.default_timer()
mapping = {
    "properties": {
        "title": {"type": "text"},
        "author": {"type": "text"},
        "bibliography": {"type": "text"},
        "body": {"type": "text"}
    }
}
create_index(es, INDEX_NAME, mapping)
indexing_success = add_documents_to_index(es, INDEX_NAME, words)
t1 = timeit.default_timer()
print("Numero de documentos indexados: ", len(words))
print(f'TEMPO DE INDEXAÇÃO = {(t1 - t0):.2f}s')

if not indexing_success:
    print("Erro durante a indexação. Verifique os documentos de entrada.")
else:
    # REALIZANDO A BUSCA
    t0 = timeit.default_timer()
    results_1 = search_results(es, INDEX_NAME, queries, list(map(len, relevant_dict.values())))
    t1 = timeit.default_timer()
    print("Numero de queries: ", len(queries))
    print(f'TEMPO DE BUSCA = {(t1 - t0):.2f}s')

    # PLOTANDO GRÁFICOS DE PRECISION E RECALL
    plot_results(results_1, relevant_dict, precision_recall_at_k, k_s=range(1, 41),
                 title='Média de Precision e Recall @k da busca 1 x k (Elasticsearch)')

# Close the Elasticsearch connection (optional)
es.close()



