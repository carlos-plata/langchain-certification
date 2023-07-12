from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

# loading env variables
load_dotenv()

# Initialize language model
llm = OpenAI(model_name="text-davinci-003", temperature=0)

# Load the summarization chain
summarize_chain = load_summarize_chain(llm)
# Enter the file path from user input
file_path= input("Enter the full path of the PDF file:\n")
# Load the document using PyPDFLoader
document_loader = PyPDFLoader(file_path=file_path)
document = document_loader.load()

# Summarize the document
summary = summarize_chain(document)
print(summary['output_text'])