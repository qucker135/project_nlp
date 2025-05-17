import os
import shutil
import timeit

# Vector RAG imports

# from langchain_community.document_loaders import PyPDFLoader
# from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
# from langchain_community.embeddings.sentence_transformer import (
#     SentenceTransformerEmbeddings,
# )
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4

# Graph RAG imports

from llama_index.core import KnowledgeGraphIndex, StorageContext, Settings
from llama_index.core import Document as LlamaIndexDocument
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from IPython.display import Markdown, display
import networkx as nx
import matplotlib.pyplot as plt
import logging
import sys
from pyvis.network import Network

# read dataset
from utils import read_mctest_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '..', 'datasets', 'mctest', 'data', 'MCTest', 'mc160.dev.txt')

data = read_mctest_dataset(DATASET_PATH)

WORKDIR = os.path.join(BASE_DIR, '..', 'workdir')

if not os.path.exists(WORKDIR):
    os.makedirs(WORKDIR)
# else:
#     print(f"Workdir {WORKDIR} already exists. Shutting down.")
#     exit(1)

REPORT_PATH = os.path.join(BASE_DIR, "reports")

if not os.path.exists(REPORT_PATH):
    os.makedirs(REPORT_PATH)

QUESTIONS_FOR_STORY = 4

times_vector = []
times_graph = []

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

model = Ollama(model="qwen2m")

prompt_template = ChatPromptTemplate([
    ("system", "Answer to the question, based only on the following context: {context}. Possible answers are: A) {A} B) {B} C) {C} D) {D}. Return just a single letter - the correct answer."),
    ("human", "{question}")
])

# with open(os.path.join(REPORT_PATH, "report_vector.txt"), "w") as f_vector:
#     
#     correct_answers = 0
# 
#     for i in range(0, len(data), QUESTIONS_FOR_STORY):
#         batch = data.iloc[i:i+QUESTIONS_FOR_STORY]
#         story = batch.iloc[0]['story']
#         questions = [batch.iloc[j]['question'] for j in range(QUESTIONS_FOR_STORY)]
#         As = [batch.iloc[j]['A'] for j in range(QUESTIONS_FOR_STORY)]
#         Bs = [batch.iloc[j]['B'] for j in range(QUESTIONS_FOR_STORY)]
#         Cs = [batch.iloc[j]['C'] for j in range(QUESTIONS_FOR_STORY)]
#         Ds = [batch.iloc[j]['D'] for j in range(QUESTIONS_FOR_STORY)]
#         good_answers = [batch.iloc[j]['good_answer'] for j in range(QUESTIONS_FOR_STORY)]
# 
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=100,
#             chunk_overlap=50,
#             length_function=len,
#             is_separator_regex=False,
#             # separators=["\n\n", "\n", "\r\n", "\r"],
#             separators=['']
#         )
# 
#         chunks = text_splitter.create_documents([story])
# 
#         PATH_CHROMA = os.path.join(WORKDIR, f"chroma_{i}")
# 
#         vector_store = Chroma(
#             collection_name=f"collection_{i}",
#             embedding_function=embeddings,
#             persist_directory=PATH_CHROMA,
#         )
# 
#         uuids = [str(uuid4()) for _ in range(len(chunks))]
# 
#         vector_store.add_documents(documents=chunks, uuids=uuids)
#     
#         for j in range(QUESTIONS_FOR_STORY):
#             print(f"Question nr: {i + j + 1}")
#             query_text = questions[j]
#             start = timeit.default_timer()
# 
#             results = vector_store.similarity_search(query_text, k=5)
# 
#             context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
#             prompt = prompt_template.format(context=context_text, question=query_text, A=As[j], B=Bs[j], C=Cs[j], D=Ds[j])
#             response = model.invoke(prompt)
# 
#             end = timeit.default_timer()
#             times_vector.append(end - start)
# 
#             print(response)
#             if response[0] == good_answers[j]:
#                 correct_answers += 1
# 
#             f_vector.write(f"Question nr: {i + j + 1}\n")
#             f_vector.write(f"Correct answer: {good_answers[j]}\n")
#             f_vector.write(f"Model's answer: {response}\n")
#             f_vector.write(f"Response time: {end - start}\n\n")
# 
#     f_vector.write(f"Average time for vector store: {sum(times_vector)/len(times_vector)}\n")
#     f_vector.write(f"Total time for vector store: {sum(times_vector)}\n")
#     f_vector.write(f"Correct answers (naive): {correct_answers}\n")

llm = LlamaIndexOllama(model="llama3", temperature=0)
model = LlamaIndexOllama(model="qwen2m", temperature=0)

Settings.llm = llm
Settings.chunk_size = 512
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
    # model_name="all-MiniLM-L6-v2"
)

with open(os.path.join(REPORT_PATH, "report_graph.txt"), "a") as f_graph:
    correct_answers = 0

    for i in range(0, len(data), QUESTIONS_FOR_STORY)[14:]:
        print(f"Story nr: {i // QUESTIONS_FOR_STORY + 1}")

        batch = data.iloc[i:i+QUESTIONS_FOR_STORY]
        story = batch.iloc[0]['story']
        questions = [batch.iloc[j]['question'] for j in range(QUESTIONS_FOR_STORY)]
        As = [batch.iloc[j]['A'] for j in range(QUESTIONS_FOR_STORY)]
        Bs = [batch.iloc[j]['B'] for j in range(QUESTIONS_FOR_STORY)]
        Cs = [batch.iloc[j]['C'] for j in range(QUESTIONS_FOR_STORY)]
        Ds = [batch.iloc[j]['D'] for j in range(QUESTIONS_FOR_STORY)]
        good_answers = [batch.iloc[j]['good_answer'] for j in range(QUESTIONS_FOR_STORY)]

        doc = LlamaIndexDocument(text=story)
        print([doc])

        graph_store = SimpleGraphStore()
        storage_context = StorageContext.from_defaults(graph_store=graph_store)

        kg_index = KnowledgeGraphIndex.from_documents(
            [doc],
            storage_context=storage_context,
            llm=llm,
            max_triplets_per_chunk=10,
            # embed_model='local',
            # include_embeddings=True,
        )

        # visualize
        g = kg_index.get_networkx_graph()
        net = Network(notebook=True, cdn_resources="in_line", directed=True)
        net.from_nx(g)
        net.show(os.path.join(REPORT_PATH, f"graph_{i // QUESTIONS_FOR_STORY + 1}.html"))

        query_engine = kg_index.as_query_engine(llm=llm, include_text=False)
        print(query_engine)

        for j in range(QUESTIONS_FOR_STORY):
            print(f"Question nr: {i + j + 1}")
            query_text = questions[j]

            prompt = f"Answer with just a single letter (A, B, C or D) - the correct answer, to the following question: {query_text} Possible answers are: A) {As[j]} B) {Bs[j]} C) {Cs[j]} D) {Ds[j]}"

            # prompt = f"Answer to the following question: {query_text} Possible answers are: A) {As[j]} B) {Bs[j]} C) {Cs[j]} D) {Ds[j]}. Return just a single letter - the correct answer."

            # prompt = f"{query_text} A) {As[j]} B) {Bs[j]} C) {Cs[j]} D) {Ds[j]}."

            print(prompt)

            start = timeit.default_timer()

            response = query_engine.query(query_text)

            end = timeit.default_timer()
            times_graph.append(end - start)

            print(response)
            # if response.text[0] == good_answers[j]:
            #     correct_answers += 1

            f_graph.write(f"Question nr: {i + j + 1}\n")
            f_graph.write(f"Correct answer: {good_answers[j]}\n")
            f_graph.write(f"Model's answer: {response}\n")
            f_graph.write(f"Response time: {end - start}\n\n")
    
    f_graph.write(f"Average time for graph store: {sum(times_graph)/len(times_graph)}\n")
    f_graph.write(f"Total time for graph store: {sum(times_graph)}\n")
    # f_graph.write(f"Correct answers (naive): {correct_answers}\n")

shutil.rmtree(WORKDIR)