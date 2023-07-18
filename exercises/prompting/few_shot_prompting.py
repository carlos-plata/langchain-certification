from dotenv import load_dotenv
from langchain import PromptTemplate, FewShotPromptTemplate, LLMChain, OpenAI

# loading env variables
load_dotenv()

# Initialize LLM
llm = OpenAI(model_name="text-davinci-003", temperature=0)

examples = [
    {"color": "red", "emotion": "passion"},
    {"color": "blue", "emotion": "serenity"},
    {"color": "green", "emotion": "tranquility"},
]

example_formatter_template = """
Color: {color}
Emotion: {emotion}\n
"""
example_prompt = PromptTemplate(
    input_variables=["color", "emotion"],
    template=example_formatter_template,
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Here are some examples of colors and the emotions associated with them:\n\n",
    suffix="\n\nNow, given a new color, identify the emotion associated with it:\n\nColor: {input}\nEmotion:",
    input_variables=["input"],
    example_separator="\n",
)

# User input data for the prompt
color = input("Enter a color:\n")

formatted_prompt = few_shot_prompt.format(input=color)

# Create the LLMChain for the prompt
chain = LLMChain(llm=llm, prompt=PromptTemplate(template=formatted_prompt, input_variables=[]))

# Run the LLMChain to get the AI-generated emotion associated with the input color
response = chain.run({})

print("Color: " + color)
print("Emotion:", response)
