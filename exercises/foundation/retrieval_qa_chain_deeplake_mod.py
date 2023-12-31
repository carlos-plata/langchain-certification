from dotenv import load_dotenv
from decouple import config
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter

# loading env variables
load_dotenv()

# get activeloop keys
ACTIVELOOP_ORG_ID=config('ACTIVELOOP_ORG_ID')
ACTIVELOOP_DATASET=config('ACTIVELOOP_DATASET')

# instantiate the LLM and embeddings models
llm = OpenAI(model="text-davinci-003", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# load the existing Deep Lake dataset and specify the embedding function
my_activeloop_org_id = ACTIVELOOP_ORG_ID
my_activeloop_dataset_name = ACTIVELOOP_DATASET
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# create new documents
texts = [
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# add documents to our Deep Lake dataset
db.add_documents(docs)

retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm, chain_type="stuff", retriever=db.as_retriever()
)

# instantiate a tool that uses the retriever
tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

# create an agent that uses the tool
agent = initialize_agent(
	tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

response = agent.run("When was Michael Jordan born?")
print(response)