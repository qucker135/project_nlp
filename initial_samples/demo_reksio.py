from langchain_community.document_loaders import PyPDFLoader
# from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4

import os

# sentences = ["This is an example sentence", "Each sentence is converted"]

PATH_DOC = os.path.join("data", "Reksio.poradnik.pdf")
PATH_CHROMA = os.path.join("chroma", "chroma")

loader = PyPDFLoader(PATH_DOC)

documents = loader.load_and_split()

documents = documents[3:]

contents = "\n".join([document.page_content for document in documents])

# print(contents)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
    # separators=["\n\n", "\n", "\r\n", "\r"],
)

chunks = text_splitter.create_documents([contents])

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="model")
# embeddings = SentenceTransformerEmbeddings("model")

# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i}: {chunk.page_content}")

vector_store = Chroma(
    collection_name="reksio",
    embedding_function=embeddings,
    persist_directory=PATH_CHROMA,
)

uuids = [str(uuid4()) for _ in range(len(chunks))]

vector_store.add_documents(documents=chunks, uuids=uuids)

model = Ollama(model="qwen2m")

while True:
    query_text = input("Enter a question: ")
    results = vector_store.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # prompt_template = ChatPromptTemplate(PROMPT, context=context_text, question=query_text)
    prompt_template = ChatPromptTemplate([
        ("system", "Answer in Polish language to the question, based only on the following context: {context}."),
        ("human", "{question}")
    ])
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("BEGIN PROMPT")
    print(prompt)
    print("END PROMPT")
    response = model.invoke(prompt)
    print("BEGIN RESPONSE")
    print(response)
    print("END RESPONSE")

# print(embeddings)
# print(embeddings.shape)
