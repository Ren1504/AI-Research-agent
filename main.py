from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableWithMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.retrievers import (
    MultiQueryRetriever,
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import json


load_dotenv()

class ResearchResponse(BaseModel):
    answer: str = Field( description="The answer to the research question.")
    confidence: float = Field( description="The confidence level of the answer.")
    sources: List[str] = Field( description="The sources used to generate the answer.") 
    key_quotes: List[str] = Field( description="Key quotes from the sources that support the answer.",default = [])

    follow_up_questions: Optional[List[str]] = Field( description="Follow-up questions that can be asked based on the answer.",default = [])



class AIResearcher:
    def __init__(self,
                 persist_directory: str = "./chroma_db",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):

        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators = ["\n\n", "\n", " ", ""]
        )


        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="research_docs")

        print(f"Vector store: {persist_directory} initialized.")
        print(f"Documents indexed: {self.vectorstore._collection.count()}")

    def add_documents(self, documents: List[Document],source_name: Optional[str] = None):
        """
        Add documents to the research database.
        """

        if source_name:
            for doc in documents:
                doc.metadata["source"] = source_name

        chunks = self.splitter.split_documents(documents)
        self.vectorstore.add_documents(chunks)

        for chunk in chunks:
            chunk.metadata["indexed_at"] = datetime.now().isoformat()

        self.vectorstore.add_documents(chunks)
        print(f"Added {len(chunks)} from {len(documents)} documents.")

        return len(chunks)

    def add_text(self, text: str, source: str , metadata: dict = None) -> int:
        """
        Add a single text string as a document
        """
        doc = Document(page_content=text, metadata={"source": source, **(metadata or {})})
        return self.add_documents([doc])

    def add_texts(self, texts: List[str], source: str) -> int:
        """
        Add multiple text strings from the same source
        """
        docs = [
            Document(page_content=t, metadata={"source": source})
            for t in texts
        ]

        return self.add_documents(docs)

    def get_document_count(self) -> int:
        """
        Get the number of documents in the database
        """
        return self.vectorstore._collection.count()

    def list_sources(self) -> List[str]:
        """
        List all unique sources in the database
        """
        results = self.vectorstore._collection.get()
        sources = set()

        for metadata in results.get("metadatas", []):
            if metadata and "source" in metadata:
                sources.add(metadata["source"])

        return list(sources)

    def _build_retriever(self) -> MultiQueryRetriever:
        """
        Build a retriever for the research database
        """
        return self.vectorstore.as_retriever(search_kwargs={"k": 4},search_type="similarity")

    def _format_docs_for_context(self,docs) -> str:
        if not docs:
            return "No relevant documents found."

        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown Source")
            content = doc.page_content
            formatted_docs.append(f"Document {i+1} (Source: {source}):\n{content}\n")

        return "\n".join(formatted_docs)

    def ask(self, question: str) -> str:
        """
        Ask a research question and get a structured response.
        """
        retriever = self._build_retriever()
        docs = retriever.invoke(question)

        context = self._format_docs_for_context(docs)

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an AI researcher. Use the provided context to answer the question.
            Only use the information in the context to answer the question. If the context does not contain enough information, say "I don't know." Do not make up answers.
            Cite the sources of your information in your answer.
            Rate your confidence: high, medium, or low."""),
            ("human", "Context:\n{context}\n\nQuestion: {question} provide a detailed answer with sources citations."),
        ])

        chain = prompt_template | self.llm | StrOutputParser()

        response = chain.invoke({"context": context, "question": question})

        return response

if __name__ == "__main__":

    import shutil
    # Clear the database for testing
    shutil.rmtree("./chroma_db", ignore_errors=True)


    researcher = AIResearcher()

    researcher.add_text("Neural networks are a type of machine learning algorithm inspired by the structure and function of the human brain.", source="Test Source 1")


    researcher.add_text("The Transformer architecture has revolutionized natural language processing by enabling models to capture long-range dependencies in text.", source="Test Source 2")
    researcher.add_text("Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties.", source="Test Source 3")

    print(f"Total chunks indexed: {researcher.get_document_count()}")
    print(f"Sources in database: {researcher.list_sources()}")

    retriever = researcher._build_retriever()
    docs = retriever.invoke("What is a neural network?")

    print(researcher._format_docs_for_context(docs))

    # for i,doc in enumerate(docs):
    #     print(f"\nDocument {i+1}:")
    #     print(f"Source: {doc.metadata.get('source')}")
    #     print(f"Content: {doc.page_content[:200]}...")  # Print first 200 characters

    q1 = "What is a neural network and how does it work?"
    q2 = "Explain the Transformer architecture and its significance in NLP."
    q3 = "Can you expand the answer to the second question with more details and examples?"

    print("\n--- Research Questions and Answers ---")

    print(f"\nQuestion: {q1}")
    answer1 = researcher.ask(q1)
    print(f"Answer: {answer1}")

    print(f"\nQuestion: {q2}")
    answer2 = researcher.ask(q2)
    print(f"Answer: {answer2}")

    print(f"\nQuestion: {q3}")
    answer3 = researcher.ask(q3)
    print(f"Answer: {answer3}")

    print("\n--- End of Research Questions and Answers ---")

    import os

    print(f"Current working directory: {os.getcwd()}")
    print(f"\nFiles on disk in ./chroma_db: {os.listdir('./chroma_db')}")

    shutil.rmtree("./chroma_db", ignore_errors=True)

    

    