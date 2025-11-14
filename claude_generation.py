import anthropic
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
import numpy as np

# It's recommended to set the API key as an environment variable

def generate_question_requests(topic, q_context , num_questions = 10
                               , rng="none", max_tokens = [10], probabilites=[1.0]):
    """
    :param q_context: context for question generation
    :param num_questions: number of questions to generate
    :param max_tokens: a list providing different max token limits for generation
    :return:
    """
     #random sampling from max_tokens

    client = anthropic.Anthropic(api_key=apikey)

    reqlist=[]

    for ii in range(num_questions):
        ttoken = max_tokens[np.random.randint(high=len(max_tokens), dtype=int)]

        reqlist.append(
            Request(
            custom_id="question_id"+str(ii),
                params=MessageCreateParamsNonStreaming(
                    model="claude-sonnet-4-5",
                    max_tokens=ttoken,
                    system=[
                              {
                                "type": "text",
                                "text": "Here is a list of Yahoo ! Answers, each separated from the previous by Answers. They respond to questions about "+topic+" "+q_context,
                                "cache_control": {"type": "ephemeral"}
                              }
                            ],
                    messages=[{
                        "role": "user",
                        "content": "Generate a a Yahoo! Answer about "+topic+". Do not provide que question nor prepend with Answer:",
                    }]
                )
            )
        )


