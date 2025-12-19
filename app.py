from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm=HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2',
    task='chat-completion',
    base_url="https://router.huggingface.co"
)
model=ChatHuggingFace(llm=llm)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

parser=StrOutputParser()
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
chain = (
    {
        "context": RunnableLambda(lambda x: x["input"])
                   | retriever
                   | RunnableLambda(format_docs),
        "input": RunnableLambda(lambda x: x["input"]),
    }
    | prompt
    | model
    | parser
)

@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = chain.invoke({"input": msg})
    print("Response : ", response)
    return str(response)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
