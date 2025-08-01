import os
import uuid
from collections import defaultdict

import chromadb
import pymupdf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

DATA_PATH = "data/"

load_dotenv(".env")

# Create clients
gemini_client = genai.Client()  # Use the API key from the environment variable `GEMINI_API_KEY`
chroma_client = chromadb.Client()

# Extract pages from documents
doc_pages = defaultdict(list)
for doc_path in os.listdir(DATA_PATH):
    if doc_path.endswith(".pdf"):
        with pymupdf.open(os.path.join(DATA_PATH, doc_path)) as doc:
            for page in doc:
                doc_pages[doc_path].append(str(page.get_text()).strip())
# TODO - Summarize or compress pages
# TODO - Filter out irrelevant content (e.g. headers, page numbers, etc.)
# TODO - Chunk documents into relevant sections (instead of using full pages)
# TODO - Add image embedding support

# Create a document collection
collection = chroma_client.create_collection(name="regulatory_docs")
collection.add(
    ids=[str(uuid.uuid4()) for doc_path in doc_pages for _ in doc_pages[doc_path]],
    documents=[page for pages in doc_pages.values() for page in pages],
    metadatas=[
        {
            "source": doc_path,
            "page": page_id + 1,
        }
        for doc_path in doc_pages
        for page_id in range(len(doc_pages[doc_path]))
    ],
)


class Tool(BaseModel):
    name: str
    description: str
    model: str

    num_tokens: int = 0

    parameters: dict = {
        "query": {
            "type": "string",
            "description": "The query and context to use.",
        },
    }
    required_parameters: list[str] = ["query"]

    def act(self, **kwargs) -> str:
        raise NotImplementedError("Must be implemented.")

    def __str__(self):
        return f"{self.name}: {self.description}"

    def id(self) -> str:
        return self.name.lower().replace(" ", "_")

    def count_tokens(self) -> int:
        return self.num_tokens


class ResponseTool(Tool):
    name: str = "Response Tool"
    description: str = "Generates a final response based on the provided query and context."
    model: str

    def act(self, query: str) -> str:
        # Generate a final response
        response = gemini_client.models.generate_content(
            model=self.model,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=f"You are a {self.name}. {self.description}\nGenerate a concise and informative response.",
            ),
        )
        self.num_tokens += response.usage_metadata.total_token_count
        return response.text


class RAGTool(Tool):
    name: str = "RAG Tool"
    description: str = "Retrieves relevant documents from the collection based on the query."
    model: str

    parameters: dict = {
        "query": {
            "type": "string",
            "description": "The query to search for in the document collection. This should be a concise question or statement describing the information needed.",
        },
        "n_results": {
            "type": "integer",
            "description": "The number of results to retrieve from the document collection.",
            "minimum": 1,
        },
    }
    required_parameters: list[str] = ["query", "n_results"]

    def act(self, query: str, n_results: int = 3) -> str:
        # Query the document collection
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        # Check validity of results
        if not results or not results["documents"]:
            return "No relevant documents found."

        # Generate a response using the retrieved documents
        results_str = "\n\n".join([f"Document source: {doc_metadata['source']}, Page num: {doc_metadata['page']}\n{doc_content}" for doc_metadata, doc_content in zip(results["metadatas"][0], results["documents"][0])])
        # TODO - Structure output to allow doc page redirections

        return results_str


class Agent(Tool):
    title: str
    tools: list[Tool] = []
    memory: list[str] = []
    max_iterations: int = 5
    max_memory: int = 10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Add default response tool
        self.tools.append(ResponseTool(model=self.model))

        # Convert tools to dict for easy access
        self.tools: dict[str, Tool] = {tool.id(): tool for tool in self.tools}

    def act(self, query: str) -> str:
        # Add query to memory
        self.memory.append(f"Query: {query}")
        self.memory = self.memory[-self.max_memory :]

        # Choose a tool based on the query
        iteration = 0
        while iteration + 1 < self.max_iterations:
            print()
            print(f"+ -------------------- +")
            print(f"| [{self.title}] {(17 - len(self.title)) * ' '} |")
            print(f"| Iteration {iteration + 1}/{self.max_iterations}        |")
            print(f"+ -------------------- +\n")

            # Add memory context to query
            context = "\n".join(self.memory)

            # Choose a tool to act
            response = gemini_client.models.generate_content(
                model=self.model,
                contents=context,
                config=types.GenerateContentConfig(
                    system_instruction=f"""
                        You are a {self.title}. {self.description}
                    
                        Choose one of the following tools: {', '.join(tool.id() for tool in self.tools.values())}. 
                        When calling a tool, always provide all required parameters. 
                        
                        Example: {{'name': 'rag_tool', 'args': {{'query': 'example', 'n_results': 3}}}}""",
                    tools=[
                        types.Tool(
                            function_declarations=[
                                {
                                    "name": tool.id(),
                                    "description": tool.description,
                                    "parameters": {
                                        "type": "object",
                                        "properties": tool.parameters,
                                        "required": tool.required_parameters,
                                    },
                                }
                                for tool in self.tools.values()
                            ]
                        ),
                    ],
                ),
            )
            self.num_tokens += response.usage_metadata.total_token_count
            iteration += 1

            # Check validity of response
            if not response.candidates:
                continue
            function_call = response.candidates[0].content.parts[0].function_call
            if not function_call:
                continue

            # Act with chosen tool
            print(f"[Tokens: {response.usage_metadata.total_token_count}] Tool: {function_call.name} {function_call.args}")
            tool = self.tools.get(function_call.name)
            tool_response = tool.act(**function_call.args)
            print(f"\n{tool_response}")

            # Add tool response to memory
            self.memory.append(f"Tool: {tool.name}, Response: {tool_response}")
            self.memory = self.memory[-self.max_memory :]
            # TODO - Add memory selection/compression/isolation (see https://rlancemartin.github.io/2025/06/23/context_engineering/)

            # Check if response is final
            if tool.id() == "response_tool":
                return tool_response

        # If max iterations reached, return final response
        return self.tools["response_tool"].act(context)

    def __str__(self):
        tools_str = [f"\n+  - {tool}" for tool in self.tools.values()]
        return f"+ {self.name}: {self.description} (with max iterations: {self.max_iterations}, max memory: {self.max_memory}){''.join(tools_str)}"

    def count_tokens(self) -> int:
        return self.num_tokens + sum(tool.count_tokens() for tool in self.tools.values())


if __name__ == "__main__":
    # TODO - General: Use prompt engineering/templates to improve agent performance

    # Define variables - #
    MODEL_NAME = "gemini-2.5-flash"
    QUERY = "What are the design control requirements for verification and validation?"
    MAX_ITERATIONS = 5
    MAX_MEMORY = 10
    # ------------------ #

    # Define agents & tools
    rag_tool = RAGTool(model=MODEL_NAME)
    rag_agent = Agent(
        title="RAG",
        name="RAG Agent",
        description="Retrieves relevant documents from the collection and generates a response.",
        model=MODEL_NAME,
        tools=[rag_tool],
        max_iterations=MAX_ITERATIONS,
        max_memory=MAX_MEMORY,
    )
    orchestrator = Agent(
        title="Orchestrator",
        name="Regulatory Assistant",
        description="Answers regulatory queries by retrieving relevant documents and generating responses.",
        model=MODEL_NAME,
        tools=[rag_agent],
        max_iterations=MAX_ITERATIONS,
        max_memory=MAX_MEMORY,
    )

    print()
    print(f"+ -------------------- +")
    print(f"| Query                |")
    print(f"+ -------------------- +\n")
    print(QUERY)

    # Act with the orchestrator agent
    response = orchestrator.act(QUERY)

    print()
    print(f"+ -------------------- +")
    print(f"| Final Response       |")
    print(f"+ -------------------- +")
    print(f"Total tokens: {orchestrator.count_tokens()}\n")
    print(response)
