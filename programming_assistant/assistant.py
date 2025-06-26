from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from tools.rag import RAG
# from tools.wiki_search import search
from tools.tavily import TavilyTools
from vectorstore.store import VectorStore

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0" 
FOLDER_PATH = "<path to code repo>"

tavilyTool = TavilyTools()
vector_store = VectorStore(model_id=EMBEDDING_MODEL_ID)
vector_store.index_folder(FOLDER_PATH)
# results = vector_store.similarity_search("What is the purpose of this project?", k=3)
# print(results)

rag = RAG(vector_store=vector_store.vectorstore, retriever=vector_store.retreiver)
rag_tool = rag.vector_search_tool

tavily_search_tool = tavilyTool.search_tool
tavil_extract_tool = tavilyTool.extract_tool
tavily_crawl_tool = tavilyTool.crawl_tool
# print(tavily_search_tool.invoke({"query": "What happened at the last wimbledon"}))

# wiki_search_tool = search
# print(wiki_search_tool("who is the current president of the united states?"))
model = init_chat_model("anthropic.claude-3-5-sonnet-20240620-v1:0", model_provider="bedrock_converse")

agent = create_react_agent(
  model=model,
  tools=[rag_tool, tavily_search_tool, tavil_extract_tool, tavily_crawl_tool],
  prompt=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
        You are a ReAct-style coding assistant focused on helping me write and extend React code for my application. You have access to the following tools:

        Available Tools:

        Internal Codebase Search

        Purpose: Search the provided React codebase to find relevant components, functions, patterns, and examples.
        Usage: Submit a natural language query to retrieve code snippets, file locations, and context from the indexed codebase.
        Best Practices:
        Always start by searching the internal codebase for relevant information or examples.
        Refer to specific files, components, or functions when possible.
        Use the structure and conventions found in the codebase when generating new code.
        Tavily Web Search / Extract / Crawl (fallback tools)

        Purpose: Retrieve or extract information from the web only if the internal codebase does not provide sufficient information.
        Usage: Use these tools to find official documentation, external examples, or best practices for React and related technologies.
        Best Practices:
        Only use these tools if the internal codebase search yields no useful results.
        Cite all external sources clearly in your response.
        Guidelines for Writing and Extending React Code:

        Do not rely on prior knowledge or assumptions—always base your answers on the codebase or cited sources.
        When adding new functionality, follow the patterns, naming conventions, and structure found in the existing code.
        If you use any code from the web, always provide the source URL as a citation.
        If none of the tools return useful information, respond:
        "I'm sorry, but none of the available tools provided sufficient information to answer this question."
        Coding Workflow:

        Thought: Consider what information or code is needed and the next steps.
        Action: Select and execute the appropriate tool (prefer internal codebase search).
        Observation: Analyze the results and determine if more information is needed.
        IMPORTANT: Repeat Thought/Action/Observation cycles as needed. Only respond to the user once you have gathered all the information you need.
        Final Answer: Provide the requested code or guidance, including relevant code snippets and citations in markdown format.
        Example Workflow:

        Question: How do I allow users to upload multiple files in chat input?

        Thought: I’ll search the codebase for existing chat conversation components to understand the implementation.
        Action: Internal Codebase Search with the query "chat conversation".
        Observation: Found several widget components in src/chat/modules/components.
        Thought: I’ll look for a relevant component, to use as a template.
        Action: Internal Codebase Search with the query "upload file".
        Observation: Found a file upload component.
        Thought: I’ll use the structure of chat component to update it for the multiple file upload.
        Final Answer: Provide a code snippet for the new widget, following the existing conventions, and cite the files used as references.
        ---

        You will now receive a coding question from the user:
        """,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ),
  name="assistant"
)

from langchain.schema import HumanMessage
inputs = {
  "messages": [
    HumanMessage(
      content="""
      As a user,
      I want to have a copy button near my prompt window that copies the entire text with one click,
      So that I can quickly copy it without having to manually highlight and copy the text.

      Technical Notes:

        Will utilize the Clipboard API for copying text
        Should include appropriate ARIA labels for accessibility (explore)
        Consider the visual placement to avoid cluttering the interface
        May need fallback methods for browsers with limited clipboard support
      """
    )
  ]
}

for s in agent.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()
    

from IPython.display import Markdown
Markdown(message.content)