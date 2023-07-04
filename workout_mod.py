from langchain.llms import OpenAI

#Needs OpenAI key saved in the “OPENAI_API_KEY” environment variable.
llm = OpenAI(model="text-davinci-003", temperature=0.9)

text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
print(llm(text))