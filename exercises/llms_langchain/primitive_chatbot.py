from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# loading env variables
load_dotenv()

prompt = PromptTemplate(template="Question: {question}\nAnswer:", input_variables=["question"])
llm = OpenAI(model_name="text-davinci-003", temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)

# Enter user input (ask a question)
question= input("Enter your question:\n")

print(chain.run(question))

