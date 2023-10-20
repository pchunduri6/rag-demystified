import openai
import tiktoken


OPENAI_PRICING = {
    'gpt-35-turbo': {'prompt': 0.0015, 'completion': 0.002},
    'gpt-35-turbo-16k': {'prompt': 0.003, 'completion': 0.004},
    'gpt-4-0613': {'prompt': 0.03, 'completion': 0.06},
    'gpt-4-32k': {'prompt': 0.06, 'completion': 0.12},
    'embedding': {'hugging_face': 0, 'text-embedding-ada-002': 0.0001}
    }


OPENAI_MODEL_CONTEXT_LENGTH = {
    'gpt-35-turbo': 4097,
    'gpt-35-turbo-16k': 16385,
    'gpt-4-0613': 8192,
    'gpt-4-32k': 32768
    }


def llm_call(model,
             function_schema=None,
             output_schema=None,
             system_prompt="You are an AI assistant that answers user questions using the context provided.",
             user_prompt="Please help me answer the following question:"):

    kwargs = {}
    if function_schema is not None:
        kwargs["functions"] = function_schema
    if output_schema is not None:
        kwargs["function_call"] = output_schema

    response = openai.ChatCompletion.create(
        engine=model,
        temperature=0,
        messages=[
            {"role": "system",
                "content": system_prompt},
            {"role": "user",
                "content": user_prompt}
        ],
        **kwargs
    )
    # print cost of call
    print("🤑 LLM call cost: $", llm_call_cost(response))
    return response


def llm_call_cost(response):
    """Returns the cost of the LLM call in dollars"""
    model = response["model"]
    usage = response["usage"]
    prompt_cost = OPENAI_PRICING[model]["prompt"]
    completion_cost = OPENAI_PRICING[model]["completion"]
    prompt_token_cost = (usage["prompt_tokens"] * prompt_cost)/1000
    completion_token_cost = (usage["completion_tokens"] * completion_cost)/1000
    return "{:.4f}".format(prompt_token_cost + completion_token_cost)


def get_num_tokens_simple(model, prompt):
    """Estimate the number of tokens in the prompt using tiktoken"""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(prompt))
    return num_tokens
