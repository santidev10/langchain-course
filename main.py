from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent""" 
    url:str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources""" 
    answer:str = Field(description="The agent's answer to query")
    sources:List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")  

llm = ChatOpenAI(model="gpt-5")
tools = [
    TavilySearch(
        max_results=3,
        search_depth="basic",
        include_domains=["linkedin.com"],
    )
]
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
    system_prompt="""
    Use the search tool at most once.
    Do not repeat similar searches.
    Return the final answer as soon as you have enough information.
    """
)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "search for 3 job postings for an ai engineer "
                        "using langchain in Austin on linkedin and list their details"
                    )
                )
            ]
        },
        config={
            "recursion_limit": 3
        }
    )

    structured = result["structured_response"]
    print(structured.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
