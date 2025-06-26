from langchain_aws import BedrockEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

class VectorStore:
  def __init__(self, model_id):
    self.embeddings = BedrockEmbeddings(model_id=model_id)
    self.vectorstore = None
    self.retreiver = None

  def index_folder(self, folder_path):
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    for root, _, files in os.walk(folder_path):
        for file in files:
            # print(file)
            if file.endswith(".tsx") or file.endswith(".ts"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Split the content into chunks
                    for chunk in splitter.split_text(content):
                        doc = Document(page_content=chunk, metadata={"file_path": file_path})
                        documents.append(doc)
    if documents:
        self.vectorstore = InMemoryVectorStore.from_documents(documents, self.embeddings)
    else:
        self.vectorstore = InMemoryVectorStore.from_documents([], self.embeddings)
    self.retreiver = self.vectorstore.as_retriever(search_kwargs={"k": 3})

  def similarity_search(self, query, k=3):
      if self.vectorstore is None:
          raise ValueError("Vector store is not initialized. Call index_folder() first.")
      return self.vectorstore.similarity_search(query, k=k)