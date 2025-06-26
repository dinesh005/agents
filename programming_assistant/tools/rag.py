class RAG:
  def __init__(self, vector_store, retriever):
    self.vector_store = vector_store
    self.retriever = retriever
    self.vector_search_tool = retriever.as_tool(
      name="vector_search",
      description="Retrieves relevant documents from the vector store based on a query.",
    )
  
  def retrieve(self, query, k=3):
    if self.vector_store is None:
      raise ValueError("Vector store is not initialized. Call index_folder() first.")
    return self.vector_store.similarity_search(query, k=k)
  
  def retrieve_with_retriever(self, query, k=3):
    if self.retriever is None:
      raise ValueError("Retriever is not initialized. Call index_folder() first.")
    return self.retriever.retrieve(query, k=k)