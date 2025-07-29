from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

load_dotenv(".env")

client = genai.Client()  # Use the API key from the environment variable `GEMINI_API_KEY`


class AgentTool(BaseModel):
    """Agent tool."""

    name: str
    description: str
    model: str

    def act(self, query: str) -> str:
        response = client.models.generate_content(
            model=self.model,
            contents=query,
        )
        return response.text


if __name__ == "__main__":
    # TODO - Define variables - #
    model_name = "gemini-2.5-flash"
    query = "What documentation is needed for a mobile app that monitors heart rate?"
    # ------------------------- #

    agent_tool = AgentTool(
        name="RegulatoryAssistant",
        description="Provides regulatory information for a given query.",
        model=model_name,
    )
    response = agent_tool.act(query)
    print(response)
