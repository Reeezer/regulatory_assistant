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
                doc_pages[doc_path].append(page.get_text())

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
        return response.text


class RAGTool(Tool):
    name: str = "RAG Tool"
    description: str = "Generates a response using the documents retrieved from the collection. Give the source and page number of the documents used."
    model: str

    parameters: dict = {
        "query": {
            "type": "string",
            "description": "The query to search for in the document collection.",
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
        response = gemini_client.models.generate_content(
            model=self.model,
            contents=results_str,
            config=types.GenerateContentConfig(
                system_instruction=f"You are a {self.name}. {self.description}",
            ),
        )

        return response.text


class Agent(Tool):
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
        print(f"Query: {query}")
        print(f"\nAgent '{self.name}': {self.description}")

        # Add query to memory
        self.memory.append(f"Query: {query}")
        self.memory = self.memory[-self.max_memory :]

        # Choose a tool based on the query
        iteration = 0
        while iteration + 1 < self.max_iterations:
            print(f"\nIteration {iteration + 1}/{self.max_iterations}")

            # Add memory context to query
            context = "\n".join(self.memory)

            # Choose a tool to act
            response = gemini_client.models.generate_content(
                model=self.model,
                contents=context,
                config=types.GenerateContentConfig(
                    system_instruction=f"You are an agent that chooses a tool to act on the query. Choose one of the following tools: {', '.join(tool.id() for tool in self.tools.values())}. When calling a tool, always provide all required parameters. Example: {{'name': 'rag_tool', 'args': {{'query': 'example', 'n_results': 3}}}}",
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
            iteration += 1

            # Check validity of response
            function_call = response.candidates[0].content.parts[0].function_call
            if not function_call:
                continue

            # Act with chosen tool
            print(f" Response: {function_call.name} {function_call.args}")
            tool = self.tools.get(function_call.name)
            tool_response = tool.act(**function_call.args)
            print(f"+- Tool response: {tool_response}")

            # Add tool response to memory
            self.memory.append(f"Tool: {tool.name}, Response: {tool_response}")
            self.memory = self.memory[-self.max_memory :]

            # Check if response is final
            if tool.id() == "response_tool":
                return tool_response

        # If max iterations reached, return final response
        return self.tools["response_tool"].act(context)

    def __str__(self):
        tools_str = [f"\n+  - {tool}" for tool in self.tools.values()]
        return f"+ {self.name}: {self.description} (with max iterations: {self.max_iterations}, max memory: {self.max_memory}){''.join(tools_str)}"


if __name__ == "__main__":
    # TODO - Define variables - #
    model_name = "gemini-2.5-flash"
    query = "What documentation is needed for a mobile app that monitors heart rate?"
    query = "What are the design control requirements for verification and validation?"
    # ------------------------- #

    # Define agents & tools
    rag_tool = RAGTool(
        model=model_name,
    )
    orchestrator = Agent(
        name="Regulatory Assistant",
        description="Answers regulatory queries by retrieving relevant documents and generating responses.",
        model=model_name,
        tools=[rag_tool],
        max_iterations=2,
        max_memory=10,
    )

    # Act with the orchestrator agent
    response = orchestrator.act(query)
    print("\n\nFinal Response:")
    print(response)
