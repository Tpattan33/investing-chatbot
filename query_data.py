import argparse
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load OPENAI_API_KEY from the .env file into the environment (read-only).
# This lets both the CLI and the web app find your key automatically.
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text: str) -> str:
    """Run one RAG query and return the answer as a string.

    This is the same logic as before, just returning the answer
    instead of printing it so the web backend can reuse it.
    """
    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory = CHROMA_PATH, embedding_function = embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k = 3)
    if len(results) == 0 or results[0][1] < 0.7:
        return "Unable to find matching results."

    # Combines relevant chunks.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Fills prompt template.
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(
        model="gpt-4o",
        temperature = 0,
    )
    response = model.invoke(prompt)
    return response.content


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    # Reuse the shared query logic and print the result.
    print(query_rag(args.query_text))

if __name__ == "__main__":
    main()