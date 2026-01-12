from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

import os

load_dotenv()

def main():
    loader = TextLoader(os.path.abspath("mediumblog1.txt"))
    document = loader.load()
    print("Splitting")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    docs = text_splitter.split_documents(document)
    print(f"Number of chunks: {len(docs)}")

    print("Embedding")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("Vectorizing and Ingestion...")
    PineconeVectorStore.from_documents(docs, embeddings, index_name=os.environ.get("INDEX_NAME"))

    print("Ingestion completed!")

if __name__ == "__main__":
    main()