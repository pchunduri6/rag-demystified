import openai


OPENAI_PRICING = {
    'gpt-35-turbo': {'prompt': 0.0015, 'completion': 0.002},
    'gpt-35-turbo-16k': {'prompt': 0.003, 'completion': 0.004},
    'gpt-4-0613': {'prompt': 0.03, 'completion': 0.06},
    'gpt-4-32k': {'prompt': 0.06, 'completion': 0.12},
    'embedding': {'hugging_face': 0, 'text-embedding-ada-002': 0.0001}
    }


def llm_call(model,
             function_schema,
             output_schema,
             system_prompt, user_prompt):

    response = openai.ChatCompletion.create(
        engine=model,
        temperature=0,
        functions=function_schema,
        function_call=output_schema,
        messages=[
            {"role": "system",
                "content": system_prompt},
            {"role": "user",
                "content": user_prompt}
        ]
    )
    return response


def llm_call_cost(response):
    """Returns the cost of the LLM call in dollars"""
    model = response["model"]
    usage = response["usage"]
    prompt_cost = OPENAI_PRICING[model]["prompt"]
    completion_cost = OPENAI_PRICING[model]["completion"]
    prompt_token_cost = (usage["prompt_tokens"] * prompt_cost)/1000
    completion_token_cost = (usage["completion_tokens"] * completion_cost)/1000
    return prompt_token_cost + completion_token_cost
