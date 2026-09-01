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
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()

class ResearchResponse(BaseModel):
    answer: str = Field( description="The answer to the research question.")
    confidence: str = Field( description="The confidence level of the answer.")
    sources: List[str] = Field( description="The sources used to generate the answer.") 
    key_quotes: List[str] = Field( description="Key quotes from the sources that support the answer.",default = [])

    follow_up_questions: Optional[List[str]] = Field( description="Follow-up questions that can be asked based on the answer.",default = [])



class AIResearcher:
    def __init__(self,
                 persist_directory: str = "./chroma_db",
                 chunk_size: int = 800,
                 chunk_overlap: int = 150):

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

        self.session_store : Dict[str, InMemoryChatMessageHistory] = {}

    def load_pdfs(self, folder_path: str) -> int:
        """
        Load all PDF files from a folder and add them to the research database.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise ValueError(f"{folder_path} is not a valid directory.")

        pdf_files = list(folder.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {folder_path}.")
            return 0

        total_chunks_added = 0
        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            chunks_added = self.add_documents(documents, source_name=pdf_file.name)
            total_chunks_added += chunks_added
            print(f"Added {chunks_added} chunks from {pdf_file.name}.")

        return total_chunks_added

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

    def _build_retriever(self,use_advanced:bool = False) -> MultiQueryRetriever:
        """
        Build a retriever for the research database
        """

        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4},search_type="similarity")

        if not use_advanced:
            return base_retriever

        multi_query_retriever = MultiQueryRetriever.from_llm(
            llm=self.llm,
            retriever=base_retriever,
        )

        return multi_query_retriever


    def _format_docs_for_context(self,docs) -> str:
        if not docs:
            return "No relevant documents found."

        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown Source")
            content = doc.page_content
            formatted_docs.append(f"Document {i+1} (Source: {source}):\n{content}\n")

        return "\n".join(formatted_docs)

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Get the chat history for a given session. If the session does not exist, create a new one.
        """
        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()
        return self.session_store[session_id]

    def ask(self, question: str, session_id: str,use_advanced: bool = True) -> str:
    
        retriever = self._build_retriever(use_advanced=use_advanced)
        docs = retriever.invoke(question)
    
        history = self._get_session_history(session_id)
    
        context = self._format_docs_for_context(docs)
    
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an AI researcher. Use the provided context to answer the question.
    
    Only use the information in the context and history to answer the question.
    If the answer cannot be found in either the history or context, say "I don't know."
    Do not make up information.
    
    Cite the sources of your information in your answer.
    Rate your confidence: high, medium, or low."""
            ),
    
            MessagesPlaceholder(variable_name="history"),
    
            (
                "human",
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Provide a detailed answer with source citations."
            ),
        ])
    
        chain = prompt_template | self.llm | StrOutputParser()
    
        response = chain.invoke({
            "context": context,
            "question": question,
            "history": history.messages[-10:]
        })
    
        history.add_message(
            HumanMessage(content=question)
        )
    
        history.add_message(
            AIMessage(content=response)
        )
    
        return response

    def ask_structured(self, question: str, session_id: str, use_advanced: bool = True) -> ResearchResponse:
        structured_llm = self.llm.with_structured_output(ResearchResponse)
        history = self._get_session_history(session_id)

        retriever = self._build_retriever(use_advanced=use_advanced)
        docs = retriever.invoke(question)
        context = self._format_docs_for_context(docs)
        sources = list(set(doc.metadata.get("source", "Unknown Source") for doc in docs))

        prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an AI researcher answering questions using a provided knowledge base.

        Rules:
        1. Use ONLY information explicitly present in the provided context or conversation history.
        2. Do NOT use your own outside knowledge.
        3. If the answer cannot be supported by the context or history, say exactly:
           "I don't know."
        4. Do not infer, assume, or invent facts.
        5. Cite the source of each factual claim when a source is available.
        6. Rate confidence as high, medium, or low based on how strongly the provided
           context/history supports the answer.
        """
    ),
    MessagesPlaceholder(variable_name="history"),
    (
        "human",
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Provide a detailed answer with source citations and a confidence rating."
    ),
])

        chain = prompt_template | structured_llm

        response = chain.invoke({
            "context": context,
            "question": question,
            "sources": sources,
            "history": (history.messages[-10:] if history.messages else [])
        })

        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=response.answer))

        return response

    def clear_session(self,session_id: str):
        """
        Clear the chat history for a given session.
        """
        if session_id in self.session_store:
            del self.session_store[session_id]
            print(f"Session {session_id} cleared.")
        else:
            print(f"Session {session_id} does not exist.")

    def get_session_history(self, session_id: str) -> List:
        """
        Get the chat history for a given session as a list of messages.
        """
        if session_id in self.session_store:
            return [
                {"role": "human", "content": msg.content} if isinstance(msg, HumanMessage) else
                {"role": "ai", "content": msg.content}
                for msg in self.session_store[session_id].messages
            ]
        return []

    def print_research_response(self,question: str, response: ResearchResponse):
        """
        Print the structured research response in a readable format.
        """
        print(f"\nQuestion: {question}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        print(f"Sources: {', '.join(response.sources)}")
        if response.key_quotes:
            print("Key Quotes:")
            for quote in response.key_quotes:
                print(f"- {quote}")
        if response.follow_up_questions:
            print("Follow-up Questions:")
            for fq in response.follow_up_questions:
                print(f"- {fq}")


if __name__ == "__main__":

    import shutil
    # Clear the database for testing
    shutil.rmtree("./chroma_db", ignore_errors=True)


    researcher = AIResearcher()

    history = researcher._get_session_history("session_1")
    # print(type(history))  # Should print <class 'langchain_core.chat_history.InMemoryChatMessageHistory'>
    # print(history.messages)  # Should print an empty list []

    # history.add_message(HumanMessage(content="Hello, I am testing the chat history."))
    # history.add_message(AIMessage(content="Hello! How can I assist you with your research today?"))
    # print(history.messages)  # Should print the two messages added above

    researcher.add_text("Neural networks are a type of machine learning algorithm inspired by the structure and function of the human brain.", source="Test Source 1")


    researcher.add_text("The Transformer architecture has revolutionized natural language processing by enabling models to capture long-range dependencies in text.", source="Test Source 2")
    researcher.add_text("Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties.", source="Test Source 3")

    # print(f"Total chunks indexed: {researcher.get_document_count()}")
    # print(f"Sources in database: {researcher.list_sources()}")

    retriever = researcher._build_retriever()
    docs = retriever.invoke("What is a neural network?")

    print(researcher._format_docs_for_context(docs))

    r = researcher.ask_structured("What is a neural network?", "session_1")

    print("\n--- Structured Research Response ---")
    print(f"Answer: {r.answer}")
    print(f"Confidence: {r.confidence}")

    if r.confidence > 0.8:
        print("\n--- High Confidence Answer ---")
        AIResearcher.print_research_response("What is a neural network?", r)
        print(f"Suggested follow-up questions: {r.follow_up_questions}")

    # for i,doc in enumerate(docs):
    #     print(f"\nDocument {i+1}:")
    #     print(f"Source: {doc.metadata.get('source')}")
    #     print(f"Content: {doc.page_content[:200]}...")  # Print first 200 characters

    # q1 = "What is NLP? Answer in one line."
    # q2 = "What is Reinforcement Learning? Answer in one line."
    # q3 = "What question was asked in the previous question?"

    # print("\n--- Research Questions and Answers ---")

    # print(f"\nQuestion: {q1}")
    # answer1 = researcher.ask(q1,"test")
    # print(f"Answer: {answer1}")

    # print(f"\nQuestion: {q3}")
    # answer2 = researcher.ask(q3,"test")
    # print(f"Answer: {answer2}")

    # print(f"\nQuestion: {q2}")
    # answer3 = researcher.ask(q2,"demo")
    # print(f"Answer: {answer3}")

    # print(f"\nQuestion: {q3}")
    # answer4 = researcher.ask(q3,"demo")
    # print(f"Answer: {answer4}")

    # print("\n--- End of Research Questions and Answers ---")

    # print("\n--- Session History ---")
    # for i,msg in enumerate(researcher.get_session_history(session_id="demo")):
    #     print(f"Message {i+1}: Role: {msg['role']}, Content: {msg['content'][:100]}.....")

    # for i,msg in enumerate(researcher.get_session_history(session_id="test")):
    #         print(f"Message {i+1}: Role: {msg['role']}, Content: {msg['content'][:100]}....")

    # print("\n--- End of Session History ---")

    # import logging 
    # logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.DEBUG)

    # retriever = researcher._build_retriever(use_advanced=True)
    # docs = retriever.invoke("What is a neural network?")
    # print(f"\nMulti-query retriever returned {len(docs)} documents.")


    # import os

    # print(f"Current working directory: {os.getcwd()}")
    # print(f"\nFiles on disk in ./chroma_db: {os.listdir('./chroma_db')}")

    # shutil.rmtree("./chroma_db", ignore_errors=True)

    

    