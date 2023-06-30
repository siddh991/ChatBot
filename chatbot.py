import os, dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

dotenv.load_dotenv()

chatbot = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0, model_name="gpt-3.5-turbo", max_tokens=50
    ), 
    chain_type="stuff", 
    retriever=FAISS.load_local("saved_embeddings", OpenAIEmbeddings())
        .as_retriever(search_type="similarity", search_kwargs={"k":5})
)

template = """
respond as succinctly as possible. {query}?
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=template,
)

print(chatbot.run(
    prompt.format(query="How many business programs are there in total?")
))