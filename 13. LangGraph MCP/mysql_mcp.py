"""Simple Airbnb MCP module."""
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage

from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

from typing_extensions import TypedDict, Annotated
import operator

# MCP GITHUB
# https://github.com/laxmimerit/MCP-Mastery-with-Claude-and-Langchain
# https://github.com/langchain-ai/langchain-mcp-adapters

# Config
LLM_MODEL = "qwen3"
BASE_URL = "http://localhost:11434"

llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL, temperature=0)

async def get_tools(): 
    client = MultiServerMCPClient(
        {
            "sqlite": {
                "command": "uvx",
                "args": [
                    "git+https://github.com/laxmimerit/mcp-server-sqlite.git",
                    "--db-path",
                    "D:\\LLM\\Agentic-RAG-with-LangGraph-and-Ollama\\11. MySQL Agent\\db\\employees_db-full-1.0.6.db"
                ],
                'transport': 'stdio'
            }
        }
    )
    tools = await client.get_tools()
    print(f"Loaded {len(tools)}")
    print(f"Tools: {tools}")

    return tools

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# create in-function agent node
async def agent(state: AgentState):
    tools = await get_tools()
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state['messages'])
    return {'messages': [response]}
    
# make sure to install uvx 
async def create_agent():
    
    tools = await get_tools()

    builder = StateGraph(AgentState)
    builder.add_node('agent', agent)
    builder.add_node('tools', ToolNode(tools))

    # add edges
    builder.add_edge(START, 'agent')
    builder.add_edge('tools', 'agent')
    builder.add_conditional_edges('agent', tools_condition)

    graph = builder.compile()
    return graph

async def search(query):
    agent = await create_agent()
    result = await agent.ainvoke({'messages': [HumanMessage(query)]})
    response = result['messages'][-1].content
    print(f"\n{response}\n")

    return response

if __name__=="__main__":
    # asyncio.run(create_agent())
    asyncio.run(search("How many employees are there?"))