# Demystifying Advanced RAG Pipelines

Retrieval-Augmented Generation (RAG) pipelines powered by large language models (LLMs) are gaining popularity for building end-to-end question answering systems. Frameworks such as [LlamaIndex](https://github.com/run-llama/llama_index) and [Haystack](https://github.com/deepset-ai/haystack) have made significant progress in making RAG pipelines easy to use. While these frameworks provide excellent abstractions for building advanced RAG pipelines, they do so at the cost of transparency. From a user perspective, it's not readily apparent what's going on under the hood, particularly when errors or inconsistencies arise. In this post, we'll shed light on the inner workings of advanced RAG pipelines by examining the mechanics, limitations, and costs that often remain opaque.


## RAG Overview

Retrieval-augmented generation (RAG) is cutting-edge AI paradigm for LLM-based question answering.
A RAG pipeline typically contains:

1. **Data Warehouse** - A collection of data sources (e.g. databases, APIs, etc.) that contain information relevant to the question answering task.

2. **Vector Retrieval** - Given a question, find the top K most similar data chunks to the question. This is done using a vector store (e.g., [Faiss](https://faiss.ai/index.html)).

3. **Response Generation** - Given the top K most similar data chunks, generate a response using a large language model (e.g. GPT-4).

RAG provides two key advantages over traditional LLM-based question answering:
1. **Up-to-date information** - The data warehouse can be updated in real-time, so the information is always up-to-date.

2. **Source tracking** -  RAG provides clear traceability, enabling users to identify the sources of information, which is crucial for accuracy verification and mitigating LLM hallucinations.

## Building advanced RAG Pipelines

To enable answering more complex questions, recent AI frameworks like LlamaIndex have introduced more advanced abstractions such as the [Sub-question Query Engine](https://gpt-index.readthedocs.io/en/latest/examples/query_engine/sub_question_query_engine.html).

In this post, we'll demystify sophisticated RAG pipelines by using the Sub-question Query Engine as an example. We'll examine the inner workings of the Sub-question Query Engine and simplify the abstractions to their core components. We'll also identify some of the challenges associated with advanced RAG pipelines.

### The setup

A data warehouse is a collection of data sources (e.g., documents, tables etc.) that contain information relevant to the question answering task.

In this example, we'll use a simple data warehouse containing multiple Wikipedia articles for different popular cities, inspired by LlamaIndex [example use-cases](https://docs.llamaindex.ai/en/stable/examples/index_structs/doc_summary/DocSummary.html). Each city's wiki is a separate data source. Note that for simplicity, we limit each document's size to fit within the LLM context limit.

Our goal is to build a system that can answer questions like:
1. *"What is the population of Chicago?"*
2. *"Give me a summary of the positive aspects of Atlanta."*
3. *"What is the city with the highest population?"*

As you can see, the questions can be simple retrieval/summarization questions over a single data source (Q1/Q2) or complex retrieval/summarization questions over multiple data sources (Q3).

We have the following *retrieval methods* at our disposal:

1. **vector retrieval** - Given a question and a data source, generate an LLM response using the top-K most similar data chunks to the question from the data source as the context. We use the off-the-shelf FAISS vector index from [EvaDB](https://github.com/georgia-tech-db/evadb) for vector retrieval. However, the concepts are applicable to any vector index.

2. **summary retrieval** - Given a summary question and a data source, generate an LLM response using the entire data source as context.

### The secret sauce

Our key insight is that each component in an advanced RAG pipeline is powered by a single LLM call. The entire pipeline is a series of LLM calls with carefully crafted prompt templates. These prompt templates are the secret sauce that enable advanced RAG pipelines to perform complex tasks.

In fact, any advanced RAG pipeline can be broken down into a series of individual LLM calls that follow a universal input pattern:

![equation](images/equation.png)

<!-- LLM input = **Prompt Template** + **Context** + **Question** -->
where:
- **Prompt Template** - A curated prompt template for the specific task (e.g. sub-question generation, summarization, etc.)
- **Context** - The context to use to perform the task (e.g. data source, top-K most similar data chunks, etc.)
- **Question** - The question to answer

Now, let's verify this principle by examining the inner workings of the Sub-question Query Engine.


### Sub-query Generation

The goal of the sub-question query engine is to take a complex question and break it down into a set of sub-questions that can be answered by a single data source and a retrieval method. For example, the question *"What is the city with the highest population?"* is broken down into five sub-questions, one for each city, of the form *"What is the population of {city}?".*

At first glance, this seems like a daunting task. Specifically, we need to answer the following questions:
1. **How do we know which sub-queries to generate?**
2. **How do we know which data source to use for each sub-query?**
3. **How do we know which retrieval method to use for each sub-query?**

Remarkably, the answer to all three questions is the same - a single LLM call! The entire sub-question query engine is powered by a single LLM call with a carefully crafted prompt template. Let's call this template the **Sub-question Prompt Template**.

```
-- Sub-question Prompt Template --

"""
    You are an AI assistant that specializes in breaking down complex questions into simpler, manageable sub-questions.
    When presented with a complex user question, your role is to generate a list of sub-questions that, when answered, will comprehensively address the original query.
    You have at your disposal a pre-defined set of functions and data sources to utilize in answering each sub-question.
    If a user question is straightforward, your task is to return the original question, identifying the appropriate function and data source to use for its solution.
    Please remember that you are limited to the provided functions and data sources, and that each sub-question should be a full question that can be answered using a single function and a single data source.
"""
```

The context for the LLM call is the names of the data sources and the functions available to the system. The question is the user question. The LLM outputs a list of sub-queries, each with a function and a data source.

For the three example questions, the LLM returns the following output:

<table>
<thead>
  <tr>
    <th>Question</th>
    <th>Subquestions</th>
    <th>Retrieval method</th>
    <th>Data Source</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>"What is the population of Chicago?"</td>
    <td>"What is the population of Chicago?"</td>
    <td>vector retrieval</td>
    <td>Chicago</td>
    </tr>
    <tr>
    <td>"Give me a summary of the positive aspects of Atlanta."</td>
    <td>"Give me a summary of the positive aspects of Atlanta."</td>
    <td>summary retrieval</td>
    <td>Atlanta</td>
    </tr>
    <tr>
    <td rowspan=5>"What is the city with the highest population?"</td>
    <td>"What is the population of Toronto?"</td>
    <td>vector retrieval</td>
    <td>Toronto</td>
    </tr>
    <tr>
    <td>"What is the population of Chicago?"</td>
    <td>vector retrieval</td>
    <td>Chicago</td>
    </tr>
    <tr>
    <td>"What is the population of Houston?"</td>
    <td>vector retrieval</td>
    <td>Houston</td>
    </tr>
    <tr>
    <td>"What is the population of Boston?"</td>
    <td>vector retrieval</td>
    <td>Boston</td>
    </tr>
    <tr>
    <td>"What is the population of Atlanta?"</td>
    <td>vector retrieval</td>
    <td>Atlanta</td>
    </tr>
</tbody>
</table>


### Vector/Summarization Retrieval

For each sub-query, we use the chosen retrieval method over the corresponding data source to retrieve the relevant information. For example, for the sub-query *"What is the population of Chicago?"*, we use vector retrieval over the Chicago data source. Similarly, for the sub-query *"Give me a summary of the positive aspects of Atlanta."*, we use summarization retrieval over the Atlanta data source.

For both retrieval methods, we use the same LLM prompt template. In fact, we find that the popular **RAG Prompt** from [LangchainHub](https://smith.langchain.com/hub) works great out-of-the-box for this step.

```
-- RAG Prompt Template --

"""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
```

Both the retrieval methods only differ in the context used for the LLM call. For vector retrieval, we use the top K most similar data chunks to the sub-query as context. For summarization retrieval, we use the entire data source as context.


### Response Aggregation

This is the final step that aggregates the responses from the sub-queries into a final response. For example, for the question *"What is the city with the highest population?"*, the sub-queries  retrieve the population of each city and then response aggregation finds and returns the city with the highest population.
The **RAG Prompt** works great for this step as well.

The context for the LLM call is the list of responses from the sub-queries. The question is the original user question and the LLM outputs a final response.


### Putting it all together

After unraveling the layers of abstraction, we uncovered the secret ingredient powering the sub-question query engine - 4 types of LLM calls each with different prompt template, context, and a question. This fits the universal input pattern that we identified earlier perfectly, and is a far cry from the complex abstractions that we started with.
The table below summarizes the entire pipeline.

![call_types_table](images/call_types_table.png)

Here is an example of the full pipeline in action for the question *"Which city with the highest population?"*.

![full_pipeline](images/simple_rag.png)

## Challenges

Now that we've demystified the inner workings of advanced RAG pipelines, let's examine some of the challenges associated with them.

1. **Query sensitivity** - The biggest challenge that we observed with these systems is the query sensitivity. The LLMs are extremely sensitive to the user question, and the pipeline fails unexpectedly for several user questions. Here are a few example failure cases that we encountered:
    - **Incorrect sub-queries** - The LLM sometimes generates incorrect sub-queries. For example, *"Which city has the highest number of tech companies?"* is broken down into *"What are the tech companies in each city?"* 5 times (once for each city) instead of *"What is the number of tech companies in Toronto?"*, *"What is the number of tech companies in Chicago?"*, etc.
    - **Incorrect retrieval method** - *"Summarize the positive aspects of Atlanta and Toronto."* results in using the vector retrieval method instead of the summarization retrieval method.

We had to put in significant effort into prompt engineering to get the pipeline to work for each question. This is a significant challenge for building robust systems.

To verify this behavior, we [implemented the example](llama_index_baseline.py) using the LlamaIndex Sub-question query engine. Consistent with our observations, the system often generates the wrong sub-queries and also uses the wrong retrieval method for the sub-queries, as shown below.

![llama_index_baseline](images/baseline.png)


2. **Cost** - The second challenge is the cost dynamics of advanced RAG pipelines. The issue is two-fold:
    - **Cost sensitivity** - The final cost of the query is dependent on the number of sub-queries generated, the retrieval method used, and the number of data sources queried. Since the LLMs are sensitive to the prompt, the cost of the query can vary significantly depending on the query and the LLM output.
    . For example, the incorrect model choice in the LlamaIndex baseline example above (`summary_tool`) results in a 3x higher cost compared to the `vector_tool` while also generating an incorrect response.
    - **Cost estimation** - Advanced abstractions in RAG frameworks obscure the estimated cost of the query. Setting up a cost monitoring system is challenging since the cost of the query is dependent on the LLM output.


## Conclusion

Advanced RAG pipelines are powerful tools for building end-to-end question answering systems. However, they are not without their challenges. In this post, we demystified the inner workings of advanced RAG pipelines and found that most advanced RAG abstractions boil down to a few LLM prompts. We also identified some of the challenges associated with advanced RAG pipelines such as prompt sensitivity, cost, and context limits. We hope that this post will help you build more robust and cost-effective RAG pipelines.



<!-- Not sure if we need these details - commenting out for now


To reliably generate the correct format of functions and data sources, we use the powerful [OpenAI function calling](https://openai.com/blog/function-calling-and-other-api-updates) feature paired with Pydantic models. We also use the [Instructor](https://github.com/jxnl/instructor) library to easily generate LLM-ready function schemas.

More details on the full schema definition can be found [here](subquestion_generator.py).

For example, the function schema to choose vector/summarization retrieval is as simple as:

```python
class FunctionEnum(str, Enum):
    """The function to use to answer the questions.
    Use vector_retrieval for factoid questions.
    Use summary_retrieval for summarization questions.
    """
    VECTOR_RETRIEVAL = "vector_retrieval"
    SUMMARY_RETRIEVAL = "summary_retrieval"
```

The data source schema definition is also straightforward:
```python
class DataSourceEnum(str, Enum):
    """The data source to use to answer the corresponding subquestion"""
    TORONTO = "Toronto"
    CHICAGO = "Chicago"
    HOUSTON = "Houston"
    BOSTON = "Boston"
    ATLANTA = "Atlanta"
```

All of this can be packaged into a simple Pydantic model:

```python
class QuestionBundle(BaseModel):
    question: str = Field(None, description="The subquestion extracted from the user's question")
    function: FunctionEnum
    data_source: DataSourceEnum
```

Using the Instructor library, we can provide the above schema as the desired output format to OpenAI.
```python
from instructor import OpenAISchema

class SubQuestionBundleList(OpenAISchema):
    subquestion_bundle_list: List[QuestionBundle] = Field(None, description="A list of subquestions - each item in the list contains a question, a function, and a data source")

response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        functions=[QuestionBundle.OpenAISchema],
        ...
)
``` -->