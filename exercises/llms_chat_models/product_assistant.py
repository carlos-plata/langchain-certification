from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# loading env variables
load_dotenv()

llm = OpenAI(model_name="text-davinci-003", temperature=0)
prompt = PromptTemplate(
  input_variables=["product"],
  template="What is a good name for a company that makes {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt)

# Enter user input (ask a question)
product= input("Enter a product:\n")

print(chain.run(product))
