from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, StorageContext, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from IPython.display import Markdown, display
import networkx as nx
import matplotlib.pyplot as plt
import logging
import sys
from pyvis.network import Network

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

documents = SimpleDirectoryReader("./llamaindex_data2").load_data()
# print(documents)

llm = Ollama(model="llama3", temperature=0)

Settings.llm = llm
Settings.chunk_size = 512
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    llm=llm,
    max_triplets_per_chunk=10,
    # embed_model='local',
    # include_embeddings=True,
)

g = kg_index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.show("example2.html")

print(graph_store)

query_engine = kg_index.as_query_engine(llm=llm, include_text=False)
# query_engine = kg_index.as_query_engine(llm=llm, include_text=False, response_mode="tree_summarize")


print(query_engine.query("Where did Todd visit? A) The city B) His mom C) The town D) The animals"))
print(query_engine.query("What did Todd say when he got home from the city? A) There were so many trees and flowers. B) There were so many people in cars. C) There's no place like home. D) There were so many animals."))
print(query_engine.query("Where does Todd live? A) The city B) with his mom C) with his dad D) In a town."))
print(query_engine.query("What did Todd see when he got to the city? A) lots of animals B) his mom C) lots of trees and flowers D) lots of people and cars"))

# ACDD

# print(query_engine.query("Who escaped from the tower?"))
# print(query_engine.query("What did the princess climb to see the castle?"))
# print(query_engine.query("What did the princess climb to see the castle? Return just a single letter - the correct answer. A) Electric pole B) mountain C) Tree D) Castle"))
# print(query_engine.query("Where does the princess live in the beginning? Return just a single letter - the correct answer. A) Castle B) house C) Cave D) High Tower"))


# triplets = graph_store.get_all_triplets()
# G = nx.DiGraph()
# 
# for t in triplets:
#     G.add_edge(t.subject, t.object, label=t.predicate)
# 
# plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(G, k=0.5)
# nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, edge_color='gray')
# nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['label'] for u, v, d in G.edges(data=True)})
# plt.title("Graf wiedzy z tekstów")
# plt.show()
