import sys
import openai
from openai.error import APIError, PermissionError, AuthenticationError, InvalidAPIType, ServiceUnavailableError
import time
from ast import literal_eval
import re
from loguru import logger
import tiktoken

from src.config import OPENAI_API_KEY, OPENAI_ORG_KEY, TOKENIZER, CHAT_COMPLETION_MODEL
from src.utils import clean_text

openai.organization = OPENAI_ORG_KEY
openai.api_key = OPENAI_API_KEY


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def split_prompt_into_chunks(prompt, chunk_size=2100):
    tokens = TOKENIZER.encode(prompt)
    chunks = []
    while len(tokens) > 0:
        current_chunk = TOKENIZER.decode(tokens[:chunk_size]).strip()
        if current_chunk:
            chunks.append(current_chunk)
        tokens = tokens[chunk_size:]
    return chunks


def generate_topics_for_text(text, topic_list):
    logger.info(f"generating keywords ... ")
    topic_modeling_prompt = f"""Analyze the following content and extract the relevant keywords from the provided TOPIC_LIST.
    The keywords should only be selected from the given TOPIC_LIST and match the content of the text.
    TOPIC_LIST = {topic_list} \n\nCONTENT: {text}
    \nBased on these guidelines:
    1. Only keywords from the TOPIC_LIST should be used.
    2. Output should be a Python list of relevant keywords from the TOPIC_LIST that describe the CONTENT.
    3. If the provided CONTENT does not contain any relevant keywords from the given TOPIC_LIST output an empty Python List ie., [].
    \nPlease provide the list of relevant topics.
    The relevant topics extracted from the provided content are: """

    time.sleep(2)
    logger.info(f"token length: {tiktoken_len(topic_modeling_prompt)}")

    response = openai.ChatCompletion.create(
        model=CHAT_COMPLETION_MODEL,
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with classifying content into specific "
                                          "topics. Your function is to extract relevant keywords from a given text, "
                                          "based on a predefined list of topics. Remember, the keywords you identify "
                                          "should only be ones that appear in the provided topic list."},
            {"role": "user", "content": f"{topic_modeling_prompt}"},
        ],
        temperature=0.0,
        max_tokens=300,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=1.0
    )
    response_str = response['choices'][0]['message']['content'].replace("\n", "").strip()
    response_str = response_str.replace("The relevant topics for the given content are: ", "").strip()
    response_str = response_str.replace("The relevant topics extracted from the provided content are: ", "").strip()
    logger.info(f"generated Keywords for this chunk: {response_str}")
    return response_str


def get_keywords_for_text(text_chunks, topic_list):
    logger.info(f"Number of chunks: {len(text_chunks)}")
    keywords_list = []
    for prompt in text_chunks:
        try:
            keywords = generate_topics_for_text(prompt, topic_list)

            if keywords == "[]" or keywords == "['']":
                continue

            # keywords = re.sub(r"(\w)'s", r'\1\'s', keywords)
            # keywords = re.sub(r"(\w)'t", r'\1\'t', keywords)
            keywords = re.sub(r"(\w)'(\w)", r'\1\'\2', keywords)

            if keywords.startswith("['") and not (keywords.endswith("']") or keywords.endswith('"]')):
                logger.warning(f"Model hallucination: {keywords}")

                if keywords.endswith("',"):
                    keywords = keywords[:-1] + "]"

                elif keywords.endswith("', '"):
                    keywords = keywords[:-3] + "]"

                elif keywords.endswith("'"):
                    keywords = keywords + "]"

                elif keywords.endswith("',..."):
                    keywords = keywords[:-4] + "]"

                elif keywords.endswith("', ...]"):
                    keywords = keywords[:-6] + "]"

                else:
                    keywords = keywords + "']"

                logger.warning(f"Keywords after fix: {keywords}")

            elif not keywords.startswith("['") and not keywords.endswith("']"):
                logger.warning(f"Elif: {keywords}")
                continue

            if isinstance(keywords, str):
                keywords = literal_eval(keywords)
            else:
                logger.warning(f"Keywords Type: {keywords}")

            keywords_list.extend(keywords)

        except openai.error.RateLimitError as rate_limit:
            logger.error(f'Rate limit error occurred: {rate_limit}')
            sys.exit(f"{rate_limit}")

        except openai.error.InvalidRequestError as invalid_req:
            logger.error(f'Invalid request error occurred: {invalid_req}')

        except (APIError, PermissionError, AuthenticationError, InvalidAPIType, ServiceUnavailableError) as ex:
            logger.error(f'Other error occurred: {str(ex)}')

    logger.success(f"Generated keywords: {keywords_list}")
    return list(set(keywords_list))


def get_primary_and_secondary_keywords(keywords_list, topic_list):
    clean_keywords_list = [clean_text(i) for i in keywords_list]
    clean_topic_list = [clean_text(i) for i in topic_list]
    primary_keywords = [i for i, j in zip(keywords_list, clean_keywords_list) if j in clean_topic_list]
    secondary_keywords = [i for i, j in zip(keywords_list, clean_keywords_list) if j not in clean_topic_list]
    logger.success(f"Primary Keywords: {len(primary_keywords)}, Secondary Keywords: {len(secondary_keywords)}")
    return primary_keywords, secondary_keywords


def apply_topic_modeling(text, topic_list):
    text_chunks = split_prompt_into_chunks(text)
    keywords_list = get_keywords_for_text(text_chunks=text_chunks, topic_list=topic_list)
    primary_keywords, secondary_keywords = get_primary_and_secondary_keywords(keywords_list=keywords_list,
                                                                              topic_list=topic_list)
    return primary_keywords, secondary_keywords
