from dotenv import load_dotenv
import os
from operator import itemgetter
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
llm = ChatOpenAI()
vectorstore = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # limit the number of results to top 3

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the context provided.
    {context}
    Question: {question}
    Provide a detailed answer: """
)

def format_docs(docs):
    """Format retrived documents into a string"""
    return "\n\n".join(doc.page_content for doc in docs) # this returns a string of the documents will go to {context}


# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# Problem will be tracing will not be under single chain so not easy to debug and understand
# ============================================================================
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


def main():
    print("Hello from langchain-ai!")

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    query = "what is Pinecone in machine learning?"
    print(f"IMPLEMENT 0: Raw LLM Call(No RAG)")
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ========================================================================
    # Option 1: Simple retrieval chain without LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"IMPLEMENT 1: Simple retrieval chain without LCEL")
    print("=" * 70)
    result_retrieval = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_retrieval)

if __name__ == "__main__":
    main()
