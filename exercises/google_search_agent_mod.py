from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper

# loading env variables
load_dotenv()

# instantiate the LLM and embeddings models
llm = OpenAI(model="text-davinci-003", temperature=0)

# Google Search via API.
search = GoogleSearchAPIWrapper()

# instantiate a tool that uses the search engine
tools = [
    Tool(
        name = "google-search",
        func=search.run,
        description="useful for when you need to search google to answer questions about current events"
    )
]

# create an agent that uses the tool
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, max_iterations=6
)

response = agent("What's the latest news about the Mars rover?")
print(response['output'])