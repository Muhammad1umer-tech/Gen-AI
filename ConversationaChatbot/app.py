from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import os
load_dotenv()
print(os.getenv("OPENAI_API_KEY"))
model = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

def test(query):
    template = """
        answer the query
        input: {input} 
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()

    response = chain.invoke({"input": query})
    print("\n\n", response) 

def load_pdf_to_database(path):
    loader = PyPDFLoader(path, allow_dangerous_deserialization=True)
    data = loader.load()
    print(len(data))

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # Number of characters per chunk
    chunk_overlap=200, # Number of overlapping characters between chunks
    )
    split_documents = text_splitter.split_documents(data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)

    # Store splits
    vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    vectorstore.save_local("../Databases/FAISS-AI_Engineer")
    print("Successfullly created the vector ") 


def test_similarity_search(query):
    vector_store = FAISS.load_local(
            "../Databases/FAISS", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search_with_score(query, k=2)
    if docs:
        print(docs)
    else:
        print("No doc found")

def retrieve_docs(query):
    vector_store = FAISS.load_local(
            "../Databases/FAISS-AI_Engineer", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(query, k=2)
    print(docs) 
    return {
        "context": docs,
        "question": query
    }
    
def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_Chain(query):
    template = """
        You are an assistant for question-answering tasks. Use the following pieces of 
        retrieved context to answer the question. If you don't know the answer, just 
        say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    retrieve_relevent_doc_according_to_question = RunnableLambda(lambda x: retrieve_docs(x['question']))
    vector_store = FAISS.load_local(
            "../Databases/FAISS-AI_Engineer", embeddings, allow_dangerous_deserialization=True)
    qa_chain = (
        {
            "context": vector_store.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    response = qa_chain.invoke(query)
    print("\n\n", response) 


def retrieval_Chain_stuff(query):
    template = """
        You are an assistant for question-answering tasks. Use the following pieces of 
        retrieved context to answer the question. If you don't know the answer, just 
        say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    stuff_chain = create_stuff_documents_chain(model, prompt)
    
    vector_store = FAISS.load_local(
            "../Databases/FAISS-AI_Engineer", embeddings, allow_dangerous_deserialization=True)
    qa_chain = (
        {
            "context": vector_store.as_retriever(),
            "question": RunnablePassthrough(),
        }
        | stuff_chain
        | StrOutputParser()
    )

    response = qa_chain.invoke(query)
    print("\n\n", response) 


# retrieval_Chain_stuff("How to study Statistics and NLP")
test("hi")
