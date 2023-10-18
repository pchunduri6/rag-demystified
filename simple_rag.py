from dotenv import load_dotenv
from subquestion_generator import SubQuestionGenerator
import evadb

if not load_dotenv():
    print(
        "Could not load .env file or it is empty. Please check if it exists and is readable."
    )
    exit(1)


if __name__ == "__main__":

    subquestion_generator = SubQuestionGenerator()

    question = "Which city has the highest population?"

    user_task = """We have a database of wikipedia articles about several cities.
                 We are building an application to answer questions about the cities."""

    subquestions = subquestion_generator.generate_subquestions(question, user_task)

    print(subquestions)
