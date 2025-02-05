import os
from dotenv import load_dotenv
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize OpenRouter model with proper configuration
model = ChatOpenAI(
    model="deepseek/deepseek-r1:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
    max_tokens=512,
    default_headers={
        "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:3000"),
        "X-Title": os.getenv("SITE_NAME", "LangGraph Agent"),
    },
    # model_kwargs was causing issues and is likely not needed for this setup.
    # Removing it for simplicity and compatibility.
    # model_kwargs={
    #     "provider": {
    #         "order": ["OpenAI", "Anthropic"],  # Providers supporting tool use
    #         "require_parameters": True
    #     }
    # }
)


# Define tools
def openrouter_qa(query: str) -> str:
    """Answers questions using OpenRouter API"""
    print(
        f"Tool 'OpenRouter QA System' executing with query: {query}"
    )  # Added print for debugging
    response = model.invoke(
        HumanMessage(content=query)
    )  # Corrected invocation with HumanMessage
    print(
        f"Tool 'OpenRouter QA System' response: {response.content}"
    )  # Added print for debugging
    return response.content


tools = [
    Tool(
        name="OpenRouter QA System",
        func=openrouter_qa,
        description="Answers questions using OpenRouter's AI models",
    )
]


# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "add_messages"]


# Create ReAct-compliant prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant with tool access. Use tools when needed."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create LangGraph agent
agent = create_react_agent(model, tools, prompt=prompt)

# Build workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")

# Configure edges - Simplified edges for ReAct agent loop
workflow.add_edge("agent", "tools")  # Agent can call tools
workflow.add_edge(
    "tools", "agent"
)  # Tools always go back to agent. Agent decides to END.
workflow.add_edge(
    "agent",
    END,
    condition=lambda state: not any(
        isinstance(message, AIMessage) and "Action:" in message.content
        for message in state["messages"][-2:-1]
    ),
)  # Agent can decide to end if no Action in last AIMessage


app = workflow.compile()


# Proper invocation with state initialization
def safe_query(question: str):
    try:
        response = app.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "input": question,  # Required for prompt template
            }
        )
        return response["messages"][-1].content
    except Exception as e:
        return f"Error processing request: {str(e)}"


# Example usage
if __name__ == "__main__":
    try:
        result = safe_query("What is the meaning of life according to AI?")
        print(f"\nFinal Answer: {result}")
    except Exception as e:
        print(f"Critical Error: {str(e)}")
