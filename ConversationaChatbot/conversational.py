from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

vector_store = FAISS.load_local(
            "../Databases/FAISS-AI_Engineer", embeddings, allow_dangerous_deserialization=True)


def retrieve_docs(query):
    vector_store = FAISS.load_local(
            "../Databases/FAISS-AI_Engineer", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(query, k=2)
    return docs
    

history_chat = [
    HumanMessage("My friend name is Umer, He is the software Engineer."),
]
condense_question_system_template = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
template = """
    You are an assistant for question-answering tasks. Use the following pieces of 
    retrieved context to answer the question. If you don't know the answer, just 
    say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

condense_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)
history_aware_chain = condense_question_prompt | model | StrOutputParser() | (lambda x: {"context": retrieve_docs(x), "question": x})
stuff_chain = create_stuff_documents_chain(model, prompt)

qa_chain = (
        history_aware_chain 
        | stuff_chain
        | StrOutputParser()
    )

# response = history_aware_chain.invoke({'input': "What is my friend name ? ", "chat_history": history_chat})
response = qa_chain.invoke({"input": "what does my friend do ?", "chat_history": history_chat})
print(response)