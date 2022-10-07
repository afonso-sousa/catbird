from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


pattern = re.compile("[\W]*")
verb_pos = ["VBZ", "VBN", "VBD", "VBP", "VB", "VBG", "IN", "TO", "PP"]
noun_pos = ["NN", "NNP", "NNS", "NNPS"]
modifier_pos = ["JJ", "FW", "JJR", "JJS", "RB", "RBR", "RBS"]


def tag(nodes, edges, question):
    porter_stemmer, stopwords_eng = PorterStemmer(), stopwords.words("english")
    ## get words and word-stems
    words, question = [node["word"].split(" ") for node in nodes], [
        w for w in question.split(" ") if len(w) > 0
    ]
    node_stem, question_stem = [
        [porter_stemmer.stem(w) for w in sent] for sent in words
    ], [porter_stemmer.stem(w) for w in question]
    ## search for the node list covering the question by each word
    question_index, node_contain = [[] for _ in question], [0 for _ in nodes]
    for index, word in enumerate(question):
        if len(word) > 0 and word not in stopwords_eng and not pattern.fullmatch(word):
            for idx, node_words in enumerate(words):
                if (
                    question_stem[index] in node_stem[idx]
                    or word in node_words
                    or any([w.count(word) > 0 for w in node_words])
                ):
                    question_index[index].append(idx)
                    node_contain[idx] += 1
    ## tag the node which covers more in the question
    for index in question_index:
        if len(index) > 0:
            index.sort(key=lambda idx: (node_contain[idx], -len(nodes[idx]["index"])))
            nodes[index[-1]]["tag"] = 1
    ## tag the node which has 'SIMILAR' edge
    #  (we assume this kind of nodes are important for asking questions)
    for index, node in enumerate(nodes):
        if "tag" not in node:
            nodes[index]["tag"] = 1 if "SIMILAR" in edges[index] else 0

    return nodes
