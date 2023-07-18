from dotenv import load_dotenv
from langchain import PromptTemplate, FewShotPromptTemplate, LLMChain, OpenAI

# loading env variables
load_dotenv()

# Initialize LLM
llm = OpenAI(model_name="text-davinci-003", temperature=0)

# User input data for the first prompt
template_question = input("Enter your first question:\n")
prompt_question = PromptTemplate(template=template_question, input_variables=[])

# Prompt 2
template_fact = """Provide a brief description of {scientist}'s work.
Answer: """
prompt_fact = PromptTemplate(input_variables=["scientist"], template=template_fact)

# Create the LLMChain for the first prompt
chain_question = LLMChain(llm=llm, prompt=prompt_question)

# Run the LLMChain for the first prompt with an empty dictionary
response_question = chain_question.run({})

# Extract the scientist's name from the response
scientist = response_question.strip()

# Create the LLMChain for the second prompt
chain_fact = LLMChain(llm=llm, prompt=prompt_fact)

# Input data for the second prompt
input_data = {"scientist": scientist}

# Run the LLMChain for the second prompt
response_fact = chain_fact.run(input_data)

print("Scientist:", scientist)
print("Fact:", response_fact)

