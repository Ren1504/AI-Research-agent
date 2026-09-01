from rag import AIResearcher


def main():

    # Create the RAG researcher
    researcher = AIResearcher(
        persist_directory="./chroma_db"
    )

    # Load and index all PDFs from the pdfs folder
    # total_chunks = researcher.load_pdfs("./pdfs")

    # print(f"\nTotal chunks indexed: {total_chunks}")

    # Start a chat session
    session_id = "pdf_chat"

    print("\n--- PDF RAG Chat ---")
    print("Type 'exit' to quit.")
    print("Type 'clear' to clear chat history.")

    while True:

        question = input("\nQuestion: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if question.lower() == "clear":
            researcher.clear_session(session_id)
            print("Chat history cleared.")
            continue

        answer = researcher.ask_structured(
            question=question,
            session_id=session_id,
            use_advanced=True
        )

        researcher.print_research_response(question, answer)


if __name__ == "__main__":
    main()