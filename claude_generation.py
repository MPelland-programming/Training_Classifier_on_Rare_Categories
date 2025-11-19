import anthropic
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
import numpy as np

# It's recommended to set the API key as an environment variable

def generate_question_requests(topic, topic_id, q_context , num_questions = 10
                               , max_tokens = [10]):
    """
    :param q_context: context for question generation
    :param num_questions: number of questions to generate
    :param max_tokens: a list providing different max token limits for generation
    :return:
    """
     #random sampling from max_tokens

    #client = anthropic.Anthropic(api_key=apikey)

    reqlist=[]

    for ii in range(num_questions):

        reqlist.append(
            Request(
            custom_id="topic"+topic_id+"_q"+str(ii),
                params=MessageCreateParamsNonStreaming(
                    model="claude-sonnet-4-5",
                    max_tokens=max_tokens,
                    system=[
                              {
                                "type": "text",
                                "text": "Here is a list of Yahoo ! Answers, each separated from the previous by 'Answers :'. They respond to questions about "+topic+" "+q_context,
                                "cache_control": {"type": "ephemeral"}
                              }
                            ],
                    messages=[{
                        "role": "user",
                        "content": "Generate a Yahoo! Answer about "+topic+". Do not provide que question nor prepend with Answer: or with a summary of the context. Simply provide an answer.",
                    }]
                )
            )
        )

    return reqlist


