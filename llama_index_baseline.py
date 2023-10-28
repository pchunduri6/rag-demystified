from pathlib import Path

import requests

from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.schema import IndexNode
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.llms import OpenAI, AzureOpenAI
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.agent import OpenAIAgent
from llama_index.embeddings import HuggingFaceEmbedding, OpenAIEmbedding
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.response_synthesizers import get_response_synthesizer
import tiktoken

api_type = ""
api_base = ""
api_version = ""
api_key = ""


embed_model_name = "hugging_face"

if embed_model_name == "hugging_face":
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-mpnet-base-v2", max_length=512
    )
elif embed_model_name == "text-embedding-ada-002":
    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name="text-embedding-ada-002",
        api_key=api_key,
        api_base=api_base,
        api_type=api_type,
        api_version=api_version,
    )

llm = AzureOpenAI(
    model="gpt-3.5-turbo",
    engine="gpt-35-turbo",
    api_key=api_key,
    api_base=api_base,
    api_type=api_type,
    api_version=api_version,
)

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

callback_manager = CallbackManager([token_counter])

service_context = ServiceContext.from_defaults(
    # system_prompt=system_prompt,
    llm=llm,
    callback_manager=callback_manager,
    embed_model=embed_model,
)


def print_token_count(token_counter, embed_model, model="gpt-35-turbo"):
    print(
        "Embedding Tokens: ",
        token_counter.total_embedding_token_count,
        "\n",
        "LLM Prompt Tokens: ",
        token_counter.prompt_llm_token_count,
        "\n",
        "LLM Completion Tokens: ",
        token_counter.completion_llm_token_count,
        "\n",
        "Total LLM Token Count: ",
        token_counter.total_llm_token_count,
        "\n",
    )
    pricing = {
        'gpt-35-turbo': {'prompt': 0.0015, 'completion': 0.002},
        'gpt-35-turbo-16k': {'prompt': 0.003, 'completion': 0.004},
        'gpt-4-0613': {'prompt': 0.03, 'completion': 0.06},
        'gpt-4-32k': {'prompt': 0.06, 'completion': 0.12},
        'embedding': {'hugging_face': 0, 'text-embedding-ada-002': 0.0001}
    }
    print(
        "Embedding Cost: ",
        pricing['embedding'][embed_model] * token_counter.total_embedding_token_count/1000,
        "\n",
        "LLM Prompt Cost: ",
        pricing[model]["prompt"] * token_counter.prompt_llm_token_count/1000,
        "\n",
        "LLM Completion Cost: ",
        pricing[model]["completion"] * token_counter.completion_llm_token_count/1000,
        "\n",
        "Total LLM Cost: ",
        pricing[model]["prompt"] * token_counter.prompt_llm_token_count/1000 + pricing[model]["completion"] * token_counter.completion_llm_token_count/1000,
        "\n",
        "Total cost: ",
        pricing['embedding'][embed_model] * token_counter.total_embedding_token_count/1000 + pricing[model]["prompt"] * token_counter.prompt_llm_token_count/1000 + pricing[model]["completion"] * token_counter.completion_llm_token_count/1000,
    )


if __name__ == "__main__":
    wiki_titles = ["Toronto", "Chicago", "Houston", "Boston", "Atlanta"]

    for title in wiki_titles:
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
    for wiki_title in wiki_titles:
        city_docs[wiki_title] = SimpleDirectoryReader(
            input_files=[f"data/{wiki_title}.txt"]
        ).load_data()

    # # Build agents dictionary
    # agents = {}

    query_engine_tools = []
    for wiki_title in wiki_titles:
        # build vector index
        vector_index = VectorStoreIndex.from_documents(
            city_docs[wiki_title], service_context=service_context
        )
        # build summary index
        summary_index = SummaryIndex.from_documents(
            city_docs[wiki_title], service_context=service_context
        )
        # define query engines
        vector_query_engine = vector_index.as_query_engine()
        list_query_engine = summary_index.as_query_engine()

        # define tools
        query_engine_tools_per_doc = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name=f"vector_tool_{wiki_title}",
                    description="Useful for questions related to specific aspects of"
                                f" {wiki_title} (e.g. the history, arts and culture,"
                                " sports, demographics, or more).",
                ),
            ),
            QueryEngineTool(
                query_engine=list_query_engine,
                metadata=ToolMetadata(
                    name=f"summary_tool_{wiki_title}",
                    description="Useful for any requests that require a holistic summary"
                                f" of EVERYTHING about {wiki_title}. For questions about"
                                " more specific sections, please use the"
                                f" vector_tool_{wiki_title}.",
                ),
            ),
        ]

        query_engine_tools.extend(query_engine_tools_per_doc)

        # build agent
        # function_llm = OpenAI(model="gpt-3.5-turbo-0613")
        # agent = OpenAIAgent.from_tools(
        #     query_engine_tools,
        #     llm=llm,
        #     verbose=True,
        # )

        # agents[wiki_title] = agent

    response_synthesizer = get_response_synthesizer(
        service_context=service_context,
        response_mode="compact",
    )

    sub_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        response_synthesizer=response_synthesizer,
        service_context=service_context,
        use_async=False,
        verbose=True,
    )

    question = "Which are the sports teams in Toronto?"
    print("Question: ", question)
    response = sub_query_engine.query(question)
    print_token_count(token_counter, embed_model_name)
