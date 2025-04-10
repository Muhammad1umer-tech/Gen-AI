from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.agent_toolkits import create_sql_agent
from langchain.schema.output_parser import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
chat = ChatOpenAI(model="gpt-4o")

db_path = './data/sqldb.db'
# Initialize the database connection
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# Set up the language model, such as OpenAI's GPT-3
llm = ChatOpenAI(temperature=0)


# Example of running a query
query = "SELECT * FROM Artist LIMIT 5"
response = db.run(query)
# print(response)


def query_to_Database(Dictquery):
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer: """)
        
    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db)
    output_from_db_chain = write_query
    response = chain.invoke({"question": query})


    chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

    chain.invoke({"question": "How many employees are there"})

agent_executor = create_sql_agent(llm, db=db, agent_type='openai-tools', verbose=True)

agent_executor.invoke(
    {
        'input': "Which country's customers spent the most?"
    }
)

