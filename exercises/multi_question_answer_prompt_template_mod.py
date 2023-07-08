from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain import HuggingFaceHub, LLMChain

# loading env variables
load_dotenv()

multi_template = """Answer the following questions one at a time.
Questions:
{questions}
Answers:
"""
long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

# initialize Hub LLM
hub_llm = HuggingFaceHub(
    repo_id='google/flan-t5-large',
    model_kwargs={'temperature':0}
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=long_prompt,
    llm=hub_llm
)

qs_str = (
    "What is the capital city of Portugal?\n" +
    "What is the largest mammal on Earth?\n" +
    "Which gas is most abundant in Earth's atmosphere?\n" +
	"What color is a ripe mango?\n"
)

# print the response
print(llm_chain.run(qs_str))

