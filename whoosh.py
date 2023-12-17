
from whoosh.fields import Schema, ID, TEXT
from whoosh import index
from whoosh.index import create_in
from whoosh.qparser import MultifieldParser, OrGroup, FuzzyTermPlugin
import os.path
import shutil
import re
import timeit
import matplotlib.pyplot as plt
import numpy as np

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

def create_index(directory, schema):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return index.create_in(directory, schema)

def add_documents_to_index(writer, documents):
    for word in documents:
        try:
            writer.add_document(index=f'{word.index}',
                                title=word.title,
                                author=word.author,
                                bibliography=word.bibliography,
                                body=word.body)
        except ValueError:
            writer.cancel()
            return False
    writer.commit()
    return True

def search_results(parser, searcher, queries, limits):
    results_dict = {}
    for i, (limit, query_to_parse) in enumerate(zip(limits, queries)):
        query = parser.parse(query_to_parse.body)
        results = searcher.search(query, limit=max(limit, 10))
        results_dict[i] = list(map(lambda x: (int(x.get('index')), x.score), results))
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
    results = np.zeros((size, 2))  # Agora é um array bidimensional para armazenar precision e recall
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

INDEX_DIRECTORY = "index_dir"
schema = Schema(index=ID(stored=True),
                title=TEXT(stored=True),
                author=TEXT(stored=True),
                bibliography=TEXT(stored=True),
                body=TEXT(analyzer=StandardAnalyzer(stoplist=None)))

# OBTENDO QUERIES, PALAVRAS E BUSCAS RELEVANTES
queries = parse_file('cran/query.txt', Query)
words = parse_file('cran/dataset.txt', Text)
relevant_dict = get_ordered_relevant_searches('cran/gabarito.txt')

# INDEXANDO RESULTADOS
t0 = timeit.default_timer()
ix = create_index(INDEX_DIRECTORY, schema)
writer = ix.writer()
indexing_success = add_documents_to_index(writer, words)
t1 = timeit.default_timer()
print("Numero de documentos: ", len(words))
print(f'TEMPO DE INDEXAÇÃO = {(t1 - t0):.2f}s')

if not indexing_success:
    print("Erro durante a indexação. Verifique os documentos de entrada.")
else:
    # REALIZANDO A BUSCA
    parser = MultifieldParser(fieldnames=["title", "author", "body"], schema=schema, group=OrGroup)
    parser.add_plugin(FuzzyTermPlugin())
    
    with ix.searcher() as searcher:
        t0 = timeit.default_timer()
        results_1 = search_results(parser, searcher, queries, list(map(len, relevant_dict.values())))
        t1 = timeit.default_timer()
        print("Numero de queries: ", len(queries))
        print(f'TEMPO DE BUSCA = {(t1 - t0):.2f}s')

        # PLOTANDO GRÁFICOS DE PRECISION E RECALL
        plot_results(results_1, relevant_dict, precision_recall_at_k, k_s=range(1, 41),
                     title='Média de Precision e Recall @k da busca 1 x k (Whoosh)')

    ix.close()
   



