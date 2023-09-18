import openai
from datetime import datetime, timedelta
from loguru import logger
import os
from dotenv import load_dotenv
import warnings
import tqdm
import pandas as pd
import traceback
import ast
import sys
import time
from openai.error import APIError, PermissionError, AuthenticationError, InvalidAPIType, ServiceUnavailableError

from src.config import ES_CLOUD_ID, ES_USERNAME, ES_PASSWORD, ES_INDEX
from src.utils import preprocess_email, ElasticSearchClient, tiktoken_len, clean_text, split_prompt_into_chunks, empty_dir

warnings.filterwarnings("ignore")
load_dotenv()

# if set to True, it will use chatgpt model ("gpt-3.5-turbo") for all the completions
CHATGPT = True

# COMPLETION_MODEL - only applicable if CHATGPT is set to False
OPENAI_ORG_KEY = os.getenv("OPENAI_ORG_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.organization = OPENAI_ORG_KEY
openai.api_key = OPENAI_API_KEY

os.makedirs("logs", exist_ok=True)
os.makedirs("gpt_output/topic_modeling", exist_ok=True)

# logs automatically rotate too big file
# empty_dir(file_path="logs")
logger.add(f"logs/es_update_topics_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log", rotation="1000 MB")


def generate_topics_for_text(text, topic_list):
    logger.info(f"generating keywords ... ")
    topic_modeling_prompt = f"""Analyze the following content and extract the relevant keywords from the provided TOPIC_LIST.
    The keywords should only be selected from the given TOPIC_LIST and match the content of the text.
    TOPIC_LIST = {topic_list} \n\nCONTENT: {text}
    \nBased on these guidelines:
    1. Only keywords from the TOPIC_LIST should be used.
    2. Output should be a Python list of relevant keywords from the TOPIC_LIST that describe the CONTENT.
    3. If the provided CONTENT does not contain any relevant keywords from the given TOPIC_LIST output an empty Python List ie., [].
    \nPlease provide the list of relevant topics:"""

    time.sleep(2)
    logger.info(f"token length: {tiktoken_len(topic_modeling_prompt)}")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
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

            # if keywords.startswith("['") and (not keywords.endswith("']") or not keywords.endswith('"]')):
            if keywords.startswith("['") and not keywords.endswith("']"):
                logger.warning(f"Model hallucination: {keywords}")

                if keywords.endswith("',"):
                    keywords = keywords[:-1] + "]"

                elif keywords.endswith("', '"):
                    keywords = keywords[:-3] + "]"

                elif keywords.endswith("'"):
                    keywords = keywords + "]"

                else:
                    keywords = keywords + "']"

                logger.warning(f"Keywords after fix: {keywords}")

            elif not keywords.startswith("['") and not keywords.endswith("']"):
                continue

            if isinstance(keywords, str):
                keywords = ast.literal_eval(keywords)
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
            # sys.exit(str(ex))

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


if __name__ == "__main__":

    delay = 3

    btc_topics_list = pd.read_csv("btc_topics.csv")
    btc_topics_list = btc_topics_list['Topics'].to_list()
    elastic_search = ElasticSearchClient(es_cloud_id=ES_CLOUD_ID, es_username=ES_USERNAME,
                                         es_password=ES_PASSWORD)

    dev_urls = [
        "https://lists.linuxfoundation.org/pipermail/bitcoin-dev/",
        # "https://lists.linuxfoundation.org/pipermail/lightning-dev/"
    ]

    for dev_url in dev_urls:
        logger.info(f"dev_url: {dev_url}")

        # if APPLY_DATE_RANGE is set to False, elasticsearch will fetch all the docs in the index
        APPLY_DATE_RANGE = False
        SAVE_CSV = True
        UPDATE_ES = False
        PREV_CSV_FILE = "tm_bd_combined_3.csv"

        if APPLY_DATE_RANGE:
            current_date_str = None
            if not current_date_str:
                current_date_str = datetime.now().strftime("%Y-%m-%d")
            start_date = datetime.now() - timedelta(days=7)
            start_date_str = start_date.strftime("%Y-%m-%d")
            logger.info(f"start_date: {start_date_str}")
            logger.info(f"current_date_str: {current_date_str}")
        else:
            start_date_str = None
            current_date_str = None

        docs_list = elastic_search.fetch_data_for_empty_keywords(ES_INDEX, dev_url, start_date_str, current_date_str)
        logger.success(f"TOTAL THREADS RECEIVED WITH AN EMPTY KEYWORDS: {len(docs_list)}")

        if docs_list:
            dev_name = dev_url.split("/")[-2]
            dataset = []

            if PREV_CSV_FILE:
                prev_df = pd.read_csv(PREV_CSV_FILE)
                logger.info(prev_df.shape)
                prev_source_ids = prev_df['source_id'].to_list()
                logger.info(len(prev_source_ids))

            for idx, doc in enumerate(tqdm.tqdm(docs_list)):
                doc_source_id = doc['_source']['id']

                if PREV_CSV_FILE:
                    if doc_source_id in prev_source_ids:
                        continue

                doc_id = doc['_id']
                doc_index = doc['_index']

                doc_body = doc['_source'].get('summary')
                if not doc_body:
                    doc_body = doc['_source'].get('body')
                    doc_body = preprocess_email(email_body=doc_body)

                logger.info(f"Doc Id: {doc_id}")
                if not doc['_source'].get('primary_topics'):
                    doc_text = ""
                    if doc_body:
                        doc_title = doc['_source'].get('title')
                        doc_text = doc_title + "\n" + doc_body

                    if doc_text:
                        try:
                            # get keywords
                            primary_kw, secondary_kw = apply_topic_modeling(text=doc_text, topic_list=btc_topics_list)

                            if SAVE_CSV and not UPDATE_ES:
                                row_data = {
                                    'primary_topics': primary_kw if primary_kw else [],
                                    'secondary_topics': secondary_kw if secondary_kw else [],
                                    'source_id': doc_source_id if doc_source_id else None
                                }
                                dataset.append(row_data)

                                if idx % 100 == 0:
                                    df = pd.DataFrame(dataset)
                                    csv_save_path = f"gpt_output/topic_modeling/topic_modeling_{dev_name}_{idx}.csv"
                                    df.to_csv(csv_save_path, index=False)
                                    logger.info(f"csv file saved at path: {csv_save_path}")
                                    time.sleep(delay)

                            elif UPDATE_ES and not SAVE_CSV:
                                if primary_kw:
                                    elastic_search.es_client.update(
                                        index=doc_index,
                                        id=doc_id,
                                        body={
                                            'doc': {
                                                "primary_topics": primary_kw
                                            }
                                        }
                                    )
                                else:
                                    elastic_search.es_client.update(
                                        index=doc_index,
                                        id=doc_id,
                                        body={
                                            'doc': {
                                                "primary_topics": []
                                            }
                                        }
                                    )
                                if secondary_kw:
                                    elastic_search.es_client.update(
                                        index=doc_index,
                                        id=doc_id,
                                        body={
                                            'doc': {
                                                "secondary_topics": secondary_kw
                                            }
                                        }
                                    )
                                else:
                                    elastic_search.es_client.update(
                                        index=doc_index,
                                        id=doc_id,
                                        body={
                                            'doc': {
                                                "secondary_topics": []
                                            }
                                        }
                                    )

                            elif SAVE_CSV and UPDATE_ES:
                                # update es index
                                if primary_kw:
                                    elastic_search.es_client.update(
                                        index=doc_index,
                                        id=doc_id,
                                        body={
                                            'doc': {
                                                "primary_topics": primary_kw
                                            }
                                        }
                                    )
                                else:
                                    elastic_search.es_client.update(
                                        index=doc_index,
                                        id=doc_id,
                                        body={
                                            'doc': {
                                                "primary_topics": []
                                            }
                                        }
                                    )
                                if secondary_kw:
                                    elastic_search.es_client.update(
                                        index=doc_index,
                                        id=doc_id,
                                        body={
                                            'doc': {
                                                "secondary_topics": secondary_kw
                                            }
                                        }
                                    )
                                else:
                                    elastic_search.es_client.update(
                                        index=doc_index,
                                        id=doc_id,
                                        body={
                                            'doc': {
                                                "secondary_topics": []
                                            }
                                        }
                                    )

                                # store in csv file
                                row_data = {
                                    'primary_topics': primary_kw if primary_kw else [],
                                    'secondary_topics': secondary_kw if secondary_kw else [],
                                    'source_id': doc_source_id if doc_source_id else None
                                }
                                dataset.append(row_data)

                                if idx % 100 == 0:
                                    df = pd.DataFrame(dataset)
                                    csv_save_path = f"gpt_output/topic_modeling/topic_modeling_{dev_name}_{idx}.csv"
                                    df.to_csv(csv_save_path, index=False)
                                    logger.info(f"csv file saved at path: {csv_save_path}")
                                    time.sleep(delay)

                            else:  # not SAVE_CSV and not UPDATE_ES
                                pass

                        except Exception as ex:
                            logger.error(f"Error: apply_topic_modeling: {traceback.format_exc()}")
                            df = pd.DataFrame(dataset)
                            csv_save_path = f"gpt_output/topic_modeling/topic_modeling_{dev_name}_exception_{idx}.csv"
                            df.to_csv(csv_save_path, index=False)
                            logger.info(f"csv file saved at path: {csv_save_path}")
                            time.sleep(delay)
                    else:
                        logger.info(f"Doc body text not found: {doc_id}")

            df = pd.DataFrame(dataset)
            csv_save_path = f"gpt_output/topic_modeling_{dev_name}_final.csv"
            df.to_csv(csv_save_path, index=False)
            logger.info(f"csv file saved at path: {csv_save_path}")
            time.sleep(delay)

        logger.success(f"Process complete for dev_url: {dev_url}")

