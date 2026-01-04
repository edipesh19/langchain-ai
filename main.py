from tabnanny import verbose

from dotenv import load_dotenv
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

@tool
def get_current_weather(location: str) -> str:
    """Get the current weather in a given location"""
    return f"The weather in {location} is sunny"

def main():
    print("Hello from langchain-ai!")
    print(os.getenv("GOOGLE_API_KEY"))
    llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.7,
        verbose="True"
)

    # Simple invocation
    messages = [
        SystemMessage(content="You are a weather expert"),
        HumanMessage(content="What is the weather in Tokyo?")
    ]

    # response = llm.invoke(messages)
    tools = [get_current_weather]
    agent = create_react_agent(llm, tools, debug=True)
    final_answer = agent.invoke({"messages": [("user", "What is the weather of capital of india?")]})
    print(f"Final Agent Answer: {final_answer}")
    print("--------------------------------")

if __name__ == "__main__":
    main()
