# Load
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import qdrant
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Qdrant

import os
import time
import arxiv

# ------------------------------------- Loader -------------------------------------------------------

dir_path = "papers"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

client = arxiv.Search(
    # 300 Characters only for Query
    query ='Neural Networks',
    max_results= 10,
    sort_by= arxiv.SortCriterion.LastUpdatedDate,
    sort_order=arxiv.SortOrder.Descending
)

for res in client.results():
    while True:
        try:
            res.download_pdf(dirpath=dir_path)
            print(f"Paper ID: {res.get_short_id()} with title: {res.title} downloaded")
            break
        except FileNotFoundError:
            print("File not found")
            break
        except arxiv.HTTPError:
            print("Forbidden")
            break
        except ConnectionError as e:
            print(f"Connection Error: {e}")
            time.sleep(5)

# ------------------------------------- End of Loader -------------------------------------------------------

# ------------------------------------- Creation of Embeddings -------------------------------------------------------
papers = []
loader = DirectoryLoader(dir_path, glob="./*.pdf", loader_cls=PyPDFLoader)
papers = loader.load()
print(f"Total papers length: {len(papers)}")

full_text = ""
for paper in papers:
    full_text = full_text + paper.page_content

# merging all text into a single text and chunk them into vector embeddings
full_text = " ".join(elements for elements in full_text.splitlines() if elements)
print(f"Length of full text: {len(full_text)}")

split_text = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunked_papers = split_text.create_documents([full_text])

# Creation of embedding and storing in vectors db
qdrant = Qdrant.from_documents(
    documents=chunked_papers,
    embedding=GPT4AllEmbeddings(),
    path="./tmp/local_qdrant",
    collection_name="arxiv_papers",
)

# ------------------------------------- Defintion for Retriever -------------------------------------------------------
retriever = qdrant.as_retriever()

# ------------------------------------- Definition of Chain and Prompt -------------------------------------------------------


# Prompt
# Optionally, pull from the Hub
# from langchain import hub
# prompt = hub.pull("rlm/rag-prompt")
# Or, define your own:
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
# Select the LLM that you downloaded
ollama_llm = "llama2:7b-chat"
model = ChatOllama(model=ollama_llm)

# RAG chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
