# # %%
# from openie import StanfordOpenIE

# # https://stanfordnlp.github.io/CoreNLP/openie.html#api
# # Default value of openie.affinity_probability_cap is 1/3.
# properties = {
#     "openie.affinity_probability_cap": 2 / 3,
# }

# with StanfordOpenIE(properties=properties) as client:
#     text = "Barack Obama was born in Hawaii. Richard Manning wrote this sentence."
#     print("Text: %s." % text)
#     for triple in client.annotate(text):
#         print("|-", triple)

#     graph_img_path = "graph.png"
#     client.generate_graphviz_graph(text, graph_img_path)
#     print("Graph generated: %s." % graph_img_path)

#     with open("pg6130.txt", encoding="utf8") as r:
#         corpus = r.read().replace("\n", " ").replace("\r", "")

#     triples_corpus = client.annotate(corpus[0:5000])
#     print("Corpus: %s [...]." % corpus[0:80])
#     print("Found %s triples in the corpus." % len(triples_corpus))
#     for triple in triples_corpus[:3]:
#         print("|-", triple)
#     print("[...]")

# # %%
# import nltk.data

# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# sentences = tokenizer.tokenize(corpus)

# # %%
# sample = ' '.join(sentences[:5])
# triples_corpus = client.annotate(' '.join(sentences[:5]))

# # %%
# client.generate_graphviz_graph(sample, 'graph2.png')

# # %%
# import stanza
# stanza.install_corenlp()
# from stanza.server import CoreNLPClient

# text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."
# with CoreNLPClient(
#         annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'],
#         timeout=30000,
#         memory='6G') as client:
#     ann = client.annotate(text)

# # %%
# # get the first sentence
# sentence = ann.sentence[0]

# # get the constituency parse of the first sentence
# constituency_parse = sentence.parseTree
# print(constituency_parse)
# # %%
# print(sentence.basicDependencies)
# # %%
# # get the first token of the first sentence
# token = sentence.token[0]
# print(token.value, token.pos, token.ner)

# # %%
# # get an entity mention from the first sentence
# print(sentence.mentions[0].entityMentionText)

# # access the coref chain in the input text
# print(ann.corefChain)

# # %%
# from stanza.server import CoreNLPClient

# text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."

# #with CoreNLPClient(annotators=["tokenize","ssplit","pos","lemma","depparse","natlog","openie"], be_quiet=False) as client:
# with CoreNLPClient(annotators=["openie"]) as client:
#     ann = client.annotate(text)
#     #print(ann)
#     for sentence in ann.sentence:
#         for triple in sentence.openieTriple:
#             print(triple)

# # %%
