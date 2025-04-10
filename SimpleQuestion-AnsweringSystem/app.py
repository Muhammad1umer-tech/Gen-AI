from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from uuid import uuid4
import faiss

load_dotenv()
model = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()

    print("Length, for testing purpose->: ", len(pages))
    return pages

def make_docs_push_to_vector_store(pages):
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    try: 
        documents = []
        for page in pages:
            document= Document(
            page_content=page.page_content, metadata=page.metadata)
            
            documents.append(document)
            uuids = [str(uuid4()) for _ in range(len(documents))]

            vector_store.add_documents(documents=documents, ids=uuids)

        vector_store.save_local("../Databases/FAISS")

        print("Successfully created vectorDatabase and save it")
    except Exception as e:
        print("Error in make_docs_push_to_vector_store function: ", str(e))


def test_similarity_search(query):
    vector_store = FAISS.load_local(
            "../Databases/FAISS", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search_with_score(query, k=2)
    if docs:
        print(docs)
    else:
        print("No doc found")

def retrieve_docs(dic):
    vector_store = FAISS.load_local(
            "../Databases/FAISS", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.asimilarity_search_with_relevance_scores(dic['query'], k=2)
    dic['docs'] = docs
    print(docs[0][1])
    return dic
    

def chaining(dic):
    template =  """You are a helpfull assistant that answer 
        the user query according to given context. Make sure, responses are short and simple.
        UserQuery: {query}
        provided document: {docs}
        """
    prompt_template = ChatPromptTemplate.from_template(template)
    docuemnt_retriever = RunnableLambda(lambda x: retrieve_docs(x))

    chain = docuemnt_retriever | prompt_template | model | StrOutputParser()
    response = chain.invoke({"query": "who are the authors of this research paper?"})

    print(response)


test_similarity_search("Tell me the author of this paper")



