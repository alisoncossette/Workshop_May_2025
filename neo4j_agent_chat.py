"""
neo4j_agent_chat.py
-------------------
A fully explained, modern conversational agent using LangChain, OpenAI, and Neo4j.

Features:
- Tracks sessions and messages in Neo4j
- Logs which company and risk factor nodes are involved in each answer
- Chains messages in order using the :NEXT relationship
- Clean, secure, and ready for extension
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_neo4j import Neo4jChatMessageHistory, Neo4jGraph
from uuid import uuid4

# ---- Configuration ----
# Ensure your .env file contains:
# NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io
# NEO4J_USERNAME=your_username
# NEO4J_PASSWORD=your_password
# OPENAI_API_KEY=your_openai_key

SESSION_ID = str(uuid4())
llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
)

# ---- Agent Prompt ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert on SEC filings and company data. Answer questions using only the information available in the graph. If you don't know, say you don't know."),
    ("human", "{input}"),
])

# ---- Tool: Query for Company Risk Factors ----
last_query_nodes = {}

def company_query_tool(query: str) -> str:
    """
    Query Neo4j for risk factors associated with a company.
    Returns company name and a list of risk factor names and IDs.
    Also saves the involved node IDs for logging.
    """
    cypher = """
    MATCH (c:Company)-[:FACES_RISK]->(r:RiskFactor)
    WHERE toLower(c.name) CONTAINS toLower($query)
    RETURN elementId(c) AS company_id, c.name AS company_name, collect({id: elementId(r), text: r.name}) AS risks
    LIMIT 1
    """
    results = graph.query(cypher, params={"query": query})
    if results:
        company_id = results[0]['company_id']
        company_name = results[0]['company_name']
        risks = results[0]['risks']
        risk_lines = "\n".join([f"- {r['text']} (id: {r['id']})" for r in risks])
        global last_query_nodes
        last_query_nodes = {
            "company_id": company_id,
            "risk_ids": [r['id'] for r in risks]
        }
        return f"Company: {company_name} (id: {company_id})\nRisk Factors:\n{risk_lines}"
    last_query_nodes.clear()
    return "No matching company or risk factors found."

# ---- Logging: Store Session, Message, and Relationships ----
def log_query_nodes_to_neo4j(session_id, question, answer, company_id, risk_ids):
    """
    Log the session, message, and involved nodes to Neo4j.
    Also chains messages in order via :NEXT relationship.
    """
    cypher = """
    MERGE (s:Session {id: $session_id})
    OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(prev:Message)
    WITH s, prev
    ORDER BY prev.timestamp DESC
    LIMIT 1
    CREATE (m:Message {question: $question, answer: $answer, timestamp: datetime()})
    MERGE (s)-[:HAS_MESSAGE]->(m)
    FOREACH (_ IN CASE WHEN prev IS NOT NULL THEN [1] ELSE [] END |
        MERGE (prev)-[:NEXT]->(m)
    )
    WITH m
    MATCH (c) WHERE elementId(c) = $company_id
    MERGE (m)-[:INVOLVES_COMPANY]->(c)
    WITH m
    UNWIND $risk_ids AS rid
    MATCH (r) WHERE elementId(r) = rid
    MERGE (m)-[:INVOLVES_RISK]->(r)
    """
    graph.query(
        cypher,
        params={
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "company_id": company_id,
            "risk_ids": risk_ids,
        }
    )

# ---- Tool Registration and Agent Setup ----
company_tool = Tool.from_function(
    name="Company Info",
    description="Query the graph for company risk factors or information. Input is a company name or question.",
    func=company_query_tool,
)
tools = [company_tool]
agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ---- Conversation Function ----
conversation = []
def ask_agent(question):
    """
    Ask the agent a question. Logs the message and involved nodes in Neo4j.
    Maintains message order within the session.
    """
    response = chat_agent.invoke(
        {"input": question},
        {"configurable": {"session_id": SESSION_ID}},
    )
    answer = response["output"]
    conversation.append({"question": question, "response": answer})
    if 'last_query_nodes' in globals() and last_query_nodes:
        log_query_nodes_to_neo4j(
            SESSION_ID,
            question,
            answer,
            last_query_nodes.get("company_id"),
            last_query_nodes.get("risk_ids", [])
        )
    print(answer)
    return answer

if __name__ == "__main__":
    print("Session ID:", SESSION_ID)
    print("Try: ask_agent('What are the risk factors for Apple?')")
