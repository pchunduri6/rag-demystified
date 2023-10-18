
import json
from typing import List, Literal
from enum import Enum

from instructor import OpenAISchema
from pydantic import BaseModel, Field
from openai_utils import llm_call, llm_call_cost


DEFAULT_SUBQUESTION_GENERATOR_PROMPT = """
                 You are an AI agent that takes a complex user question and returns a list of simple subquestions to answer the user's question.
                 You will also return the function and data source to use to answer each subquestion.
                 You can only use the provided functions and data sources.
                 The subquestions should be complete questions that can be answered by a single function and a single data source.
                 If the user question is simple and can be answered directly by a single data source, just return the user question and the data source to use.
                 """

DEFAULT_USER_TASK = """
                Please generate a list of subquestions to answer the user's question.
                """


class DataSource(str, Enum):
    """The data source to use to answer the corresponding subquestion"""
    TORONTO = "toronto"
    CHICAGO = "chicago"
    HOUSTON = "houston"
    BOSTON = "boston"
    ATLANTA = "atlanta"


class Function(BaseModel):
    name: Literal["vector_retrieval", "llm_retrieval"] = Field(description="""The function to use to answer the questions.
                                                                        Use vector_retrieval for factoid and specific context-based questions.
                                                                        Use llm_retrieval for summarization questions.""")


class SubQuestion(BaseModel):
    subquestion: str = Field(None, description="The subquestion extracted from the user's question")
    function: Function  # = Field(None, description="The function to use to answer the corresponding subquery.")
    data_source: DataSource  # = Field(None, description="The data source to use to answer the corresponding subquery")


class SubQuestionsList(OpenAISchema):
    subquestions: List[SubQuestion] = Field(None, description="A list of subquestions - each item in the list contains a question, a function, and a data source")


class SubQuestionGenerator:
    """Generates a list of subquestions from a user question.
    """
    def __init__(self, data_sources: List[str] = None, functions: List[str] = None):
        self.data_sources = data_sources
        self.functions = functions

    def generate_subquestions(self,
                              question,
                              system_prompt=DEFAULT_SUBQUESTION_GENERATOR_PROMPT,
                              user_task=DEFAULT_USER_TASK,
                              ) -> SubQuestionsList:
        """Generates a list of subquestions from a user question along with the
        data source and the function to use to answer the question using OpenAI LLM.
        """

        user_prompt = f"{user_task}\n Here is the user question: {question}"

        response = llm_call(model="gpt-4-0613",
                            function_schema=[SubQuestionsList.openai_schema],
                            output_schema={"name": SubQuestionsList.openai_schema["name"]},
                            system_prompt=system_prompt,
                            user_prompt=user_prompt)

        price = llm_call_cost(response)
        print("🤑 LLM call cost: ", price)
        subquestions = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])

        subquestions = SubQuestionsList(**subquestions)
        return subquestions
