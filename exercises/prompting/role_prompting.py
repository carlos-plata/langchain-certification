from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain, OpenAI

# loading env variables
load_dotenv()

llm = OpenAI(model_name="text-davinci-003", temperature=0)

template = """
As a futuristic robot band conductor, I need you to help me come up with a song title.
What's a cool song title for a song about {theme} in the year {year}?
"""

prompt = PromptTemplate(
    input_variables=["theme", "year"],
    template=template,
)

# User input data for the prompt
theme = input("Enter a theme:\n")
year = input("Enter a year:\n")

# Input data for the prompt
input_data = {"theme": theme, "year": year}

chain = LLMChain(llm=llm, prompt=prompt)

# Run the LLMChain to get the AI-generated song title
response = chain.run(input_data)

print("Theme:" + theme)
print("Year:" + year)
print("AI-generated song title:", response)
