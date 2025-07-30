from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv(".env")

client = genai.Client()  # Use the API key from the environment variable `GEMINI_API_KEY`


class Tool(BaseModel):
    name: str
    description: str
    model: str

    def act(self, query: str) -> str:
        raise NotImplementedError("Must be implemented.")

    def __str__(self):
        return f"{self.name}: {self.description}"

    def id(self) -> str:
        return self.name.lower().replace(" ", "_")


class ResponseTool(Tool):
    name: str = "Response Tool"
    description: str = "Generates a final response based on the provided query."
    model: str

    def act(self, query: str) -> str:
        response = client.models.generate_content(
            model=self.model,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=f"You are a {self.name}. {self.description}\nGenerate a concise and informative response.",
            ),
        )
        return response.text


class RAGTool(Tool):
    name: str = "RAG Tool"
    description: str = "Generates a response using a document retrieval system."
    model: str

    def act(self, query: str) -> str:
        return "No relevant documentation found for the query."


class Agent(Tool):
    tools: list[Tool] = []
    max_iterations: int = 5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Add default response tool
        self.tools.append(ResponseTool(model=self.model))

        # Convert tools to dict for easy access
        self.tools: dict[str, Tool] = {tool.id(): tool for tool in self.tools}

    def act(self, query: str) -> str:
        print(f"Agent '{self.name}': {self.description}")
        print(f"Query: {query}")
        print()

        # Choose a tool based on the query
        iteration = 0
        while iteration + 1 < self.max_iterations:
            print(f"Iteration {iteration + 1}/{self.max_iterations}")

            # Choose a tool to act
            response = client.models.generate_content(
                model=self.model,
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=f"You are an agent that chooses a tool to act on the query. Choose one of the following tools: {', '.join(tool.id() for tool in self.tools.values())}.",
                    tools=[
                        types.Tool(
                            function_declarations=[
                                {
                                    "name": tool.id(),
                                    "description": tool.description,
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "query": {
                                                "type": "string",
                                                "description": "The query to use.",
                                            },
                                        },
                                        "required": ["query"],
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
            tool_name = function_call.name
            tool_args = function_call.args
            tool = self.tools.get(tool_name)
            print(f"- Using tool: {tool.name}")
            tool_response = tool.act(**tool_args)

            # Check if response is final
            if tool.id() == "response_tool":
                return tool_response

            # Update query for next iteration
            query = tool_response

        # If max iterations reached, return response tool's act
        return self.tools["response_tool"].act(query)

    def __str__(self):
        tools_str = [f"\n+  - {tool}" for tool in self.tools.values()]
        return f"+ {self.name}: {self.description} (with max iterations: {self.max_iterations}){''.join(tools_str)}"


if __name__ == "__main__":
    # TODO - Define variables - #
    model_name = "gemini-2.5-flash"
    query = "What documentation is needed for a mobile app that monitors heart rate?"
    # ------------------------- #

    # Define agents & tools
    rag_tool = RAGTool(
        model=model_name,
    )
    orchestrator = Agent(
        name="Regulatory Assistant",
        description="Finds the documentation requirements for a given query.",
        model=model_name,
        tools=[rag_tool],
        max_iterations=5,
    )

    # Act with the orchestrator agent
    response = orchestrator.act(query)
    print(response)
