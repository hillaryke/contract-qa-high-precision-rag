from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

def get_answer(query):
    # This is a test answer. Replace this with your own answer.
    answer = "Hello World! This is a test answer."
    return answer