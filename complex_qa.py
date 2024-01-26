import os
from dotenv import load_dotenv
from pathlib import Path
import requests

import warnings
warnings.filterwarnings("ignore")

from subquestion_generator import generate_subquestions
import evadb
from openai_utils import llm_call


if not load_dotenv():
    print(
        "Could not load .env file or it is empty. Please check if it exists and is readable."
    )
    exit(1)


def generate_vector_stores(cursor, docs):
    """Generate a vector store for the docs using evadb.
    """
    for doc in docs:
        print(f"Creating vector store for {doc}...")
        cursor.query(f"DROP TABLE IF EXISTS {doc};").df()
        cursor.query(f"LOAD DOCUMENT 'data/{doc}.txt' INTO {doc};").df()
        evadb_path = os.path.dirname(evadb.__file__)
        cursor.query(
            f"""CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
            IMPL  '{evadb_path}/functions/sentence_feature_extractor.py';
            """).df()

        cursor.query(
            f"""CREATE TABLE IF NOT EXISTS {doc}_features AS
            SELECT SentenceFeatureExtractor(data), data FROM {doc};"""
        ).df()

        cursor.query(
            f"CREATE INDEX IF NOT EXISTS {doc}_index ON {doc}_features (features) USING FAISS;"
        ).df()
        print(f"Successfully created vector store for {doc}.")


def vector_retrieval(cursor, llm_model, question, doc_name):
    """Returns the answer to a factoid question using vector retrieval.
    """
    res_batch = cursor.query(
        f"""SELECT data FROM {doc_name}_features
        ORDER BY Similarity(SentenceFeatureExtractor('{question}'),features)
        LIMIT 3;"""
    ).df()
    context_list = []
    for i in range(len(res_batch)):
        context_list.append(res_batch["data"][i])
    context = "\n".join(context_list)
    user_prompt = f"""You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question.
                If you don't know the answer, just say that you don't know.
                Use three sentences maximum and keep the answer concise.
                Question: {question}
                Context: {context}
                Answer:"""

    response, cost = llm_call(model=llm_model, user_prompt=user_prompt)

    answer = response.choices[0].message.content
    return answer, cost


def summary_retrieval(llm_model, question, doc):
    """Returns the answer to a summarization question over the document using summary retrieval.
    """
    # context_length = OPENAI_MODEL_CONTEXT_LENGTH[llm_model]
    # total_tokens = get_num_tokens_simple(llm_model, wiki_docs[doc])
    user_prompt = f"""Here is some context: {doc}
                Use only the provided context to answer the question.
                Here is the question: {question}"""

    response, cost = llm_call(model=llm_model, user_prompt=user_prompt)
    answer = response.choices[0].message.content
    return answer, cost
    # load max of context_length tokens from the document


def response_aggregator(llm_model, question, responses):
    """Aggregates the responses from the subquestions to generate the final response.
    """
    print("-------> ⭐ Aggregating responses...")
    system_prompt = """You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question.
                If you don't know the answer, just say that you don't know.
                Use three sentences maximum and keep the answer concise."""

    context = ""
    for i, response in enumerate(responses):
        context += f"\n{response}"

    user_prompt = f"""Question: {question}
                      Context: {context}
                      Answer:"""

    response, cost = llm_call(model=llm_model, system_prompt=system_prompt, user_prompt=user_prompt)
    answer = response.choices[0].message.content
    return answer, cost


def load_wiki_pages(page_titles=["Toronto", "Chicago", "Houston", "Boston", "Atlanta"]):

    # Download all wiki documents
    for title in page_titles:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                # 'exintro': True,
                "explaintext": True,
            },
        ).json()
        page = next(iter(response["query"]["pages"].values()))
        wiki_text = page["extract"]

        data_path = Path("data")
        if not data_path.exists():
            Path.mkdir(data_path)

        with open(data_path / f"{title}.txt", "w") as fp:
            fp.write(wiki_text)

    # Load all wiki documents
    city_docs = {}
    for wiki_title in page_titles:
        input_text = open(f"data/{wiki_title}.txt", "r").read()
        city_docs[wiki_title] = input_text[:10000]
    return city_docs


if __name__ == "__main__":

    # establish evadb api cursor
    print("⏳ Connect to EvaDB...")
    cursor = evadb.connect().cursor()
    print("✅ Connected to EvaDB...")

    doc_names = ["Toronto", "Chicago", "Houston", "Boston", "Atlanta"]
    wiki_docs = load_wiki_pages(page_titles=doc_names)

    question = "Which city has the highest population?"

    user_task = """We have a database of wikipedia articles about several cities.
                 We are building an application to answer questions about the cities."""

    vector_stores = generate_vector_stores(cursor, wiki_docs)

    llm_model = "gpt-3.5-turbo"
    total_cost = 0
    while True:
        question_cost = 0
        # Get question from user
        question = str(input("Question (enter 'exit' to exit): "))
        if question.lower() == "exit":
            break
        print("🧠 Generating subquestions...")
        subquestions_bundle_list, cost = generate_subquestions(question=question,
                                                               file_names=doc_names,
                                                               user_task=user_task,
                                                               llm_model=llm_model)
        question_cost += cost
        responses = []
        for q_no, item in enumerate(subquestions_bundle_list):
            subquestion = item.question
            selected_func = item.function.value
            selected_doc = item.file_name.value
            print(f"\n-------> 🤔 Processing subquestion #{q_no+1}: {subquestion} | function: {selected_func} | data source: {selected_doc}")
            if selected_func == "vector_retrieval":
                response, cost = vector_retrieval(cursor, llm_model, subquestion, selected_doc)
            elif selected_func == "llm_retrieval":
                response, cost = summary_retrieval(llm_model, subquestion, wiki_docs[selected_doc])
            else:
                print(f"\nCould not process subquestion: {subquestion} function: {selected_func} data source: {selected_doc}\n")
                exit(0)
            print(f"✅ Response #{q_no+1}: {response}")
            responses.append(response)
            question_cost += cost

        aggregated_response, cost = response_aggregator(llm_model, question, responses)
        question_cost += cost
        print(f"\n✅ Final response: {aggregated_response}")
        print(f"🤑 Total cost for the question: ${question_cost:.4f}")
        total_cost += question_cost

    print(f"Total cost for all questions: ${total_cost:.4f}")
