from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain import HuggingFaceHub, LLMChain

# loading env variables
load_dotenv()

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
        template=template,
    input_variables=['question']
)

# user question
question = "What is the capital city of France?"

# initialize Hub LLM
hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-large',
    model_kwargs={'temperature':0}
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# ask the user question about the capital of France
print(llm_chain.run(question))
