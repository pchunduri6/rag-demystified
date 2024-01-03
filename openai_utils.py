import os
import sys
import logging

from openai import OpenAI
client = OpenAI()
import tiktoken

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    after_log,
)  # for exponential backoff

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_PRICING = {
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
    "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
    "embedding": {"hugging_face": 0, "text-embedding-ada-002": 0.0001},
}


OPENAI_MODEL_CONTEXT_LENGTH = {
    "gpt-3.5-turbo": 4097,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
}


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    after=after_log(logger, logging.INFO),
)
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def llm_call_cost(response):
    """Returns the cost of the LLM call in dollars"""
    model = response.model
    usage = response.usage
    prompt_cost = OPENAI_PRICING[model]["prompt"]
    completion_cost = OPENAI_PRICING[model]["completion"]
    prompt_token_cost = (usage.prompt_tokens * prompt_cost) / 1000
    completion_token_cost = (usage.completion_tokens * completion_cost) / 1000
    return prompt_token_cost + completion_token_cost


def llm_call(
    model,
    function_schema=None,
    output_schema=None,
    system_prompt="You are an AI assistant that answers user questions using the context provided.",
    user_prompt="Please help me answer the following question:",
    few_shot_examples=None,
):
    kwargs = {}
    if function_schema is not None:
        kwargs["functions"] = function_schema
    if output_schema is not None:
        kwargs["function_call"] = output_schema

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    if few_shot_examples is not None:
        messages.extend(few_shot_examples)
    if user_prompt is not None:
        messages.append({"role": "user", "content": user_prompt})

    response = completion_with_backoff(
        model=model,
        temperature=0,
        messages=messages,
        **kwargs
    )

    # print cost of call
    call_cost = llm_call_cost(response)
    print(f"🤑 LLM call cost: ${call_cost:.4f}")
    return response, call_cost


def get_num_tokens_simple(model, prompt):
    """Estimate the number of tokens in the prompt using tiktoken"""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(prompt))
    return num_tokens
