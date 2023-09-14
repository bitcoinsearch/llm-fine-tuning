import os
import re
import shutil

from dateutil.parser import parse
from elasticsearch import Elasticsearch
import time
from loguru import logger
from dotenv import load_dotenv
import warnings
import tiktoken
from src.config import TOKENIZER, ES_DATA_FETCH_SIZE

warnings.filterwarnings("ignore")
load_dotenv()


class ElasticSearchClient:
    def __init__(self, es_cloud_id, es_username, es_password, es_data_fetch_size=ES_DATA_FETCH_SIZE) -> None:
        self._es_cloud_id = es_cloud_id
        self._es_username = es_username
        self._es_password = es_password
        self._es_data_fetch_size = es_data_fetch_size
        self._es_client = Elasticsearch(
            cloud_id=self._es_cloud_id,
            http_auth=(self._es_username, self._es_password),
        )

    # def fetch_data_based_on_empty_keywords(self, es_index, start_date_str, current_date_str):
    #     logger.info(f"fetching the data based on empty keywords ... ")
    #     output_list = []
    #     start_time = time.time()
    #
    #     if self._es_client.ping():
    #         logger.success("connected to the ElasticSearch")
    #
    #         if start_date_str and current_date_str:
    #             query = {
    #                 "query": {
    #                     "bool": {
    #                         "must": [
    #                             {
    #                                 "range": {
    #                                     "created_at": {
    #                                         "gte": f"{start_date_str}T00:00:00.000Z",
    #                                         "lte": f"{current_date_str}T23:59:59.999Z"
    #                                     }
    #                                 }
    #                             }
    #                         ],
    #                         "must_not": {
    #                             "exists": {
    #                                 "field": "primary_topics"
    #                             }
    #                         }
    #                     }
    #                 }
    #             }
    #         else:
    #             query = {
    #                 "query": {
    #                     "bool": {
    #                         "must_not": {
    #                             "exists": {
    #                                 "field": "primary_topics"
    #                             }
    #                         }
    #                     }
    #                 }
    #             }
    #
    #         # Initialize the scroll
    #         scroll_response = self._es_client.search(index=es_index, body=query, size=self._es_data_fetch_size,
    #                                                  scroll='1m')
    #         scroll_id = scroll_response['_scroll_id']
    #         results = scroll_response['hits']['hits']
    #
    #         # Dump the documents into the json file
    #         logger.info(f"Starting dumping of {es_index} data in json...")
    #         while len(results) > 0:
    #             # Save the current batch of results
    #             for result in results:
    #                 output_list.append(result)
    #
    #             # Fetch the next batch of results
    #             scroll_response = self._es_client.scroll(scroll_id=scroll_id, scroll='1m')
    #             scroll_id = scroll_response['_scroll_id']
    #             results = scroll_response['hits']['hits']
    #
    #         logger.info(
    #             f"Dumping of {es_index} data in json has completed and has taken {time.time() - start_time:.2f} seconds.")
    #
    #         return output_list
    #     else:
    #         logger.warning('Could not connect to Elasticsearch')
    #         return output_list

    def fetch_data_for_empty_keywords(self, es_index, url=None, start_date_str=None, current_date_str=None):
        logger.info(f"fetching the data based on empty keywords ... ")
        output_list = []
        start_time = time.time()

        if self._es_client.ping():
            logger.success("connected to the ElasticSearch")

            if url and start_date_str and current_date_str:
                logger.info(f"Url: {url}, Start Date: {start_date_str}, Current Date: {current_date_str}")
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "created_at": {
                                            "gte": f"{start_date_str}T00:00:00.000Z",
                                            "lte": f"{current_date_str}T23:59:59.999Z"
                                        }
                                    }
                                },
                                {
                                    "prefix": {
                                        "domain.keyword": str(url)
                                    }
                                }
                            ],
                            "must_not": {
                                "exists": {
                                    "field": "primary_topics"
                                }
                            }
                        }
                    }
                }
            elif url and not start_date_str and not current_date_str:
                logger.info(f"Url: {url}, Start Date: {start_date_str}, Current Date: {current_date_str}")
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "prefix": {
                                        "domain.keyword": str(url)
                                    }
                                }
                            ],
                            "must_not": {
                                "exists": {
                                    "field": "primary_topics"
                                }
                            }
                        }
                    }
                }
            elif not url and start_date_str and current_date_str:
                logger.info(f"Url: {url}, Start Date: {start_date_str}, Current Date: {current_date_str}")
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "created_at": {
                                            "gte": f"{start_date_str}T00:00:00.000Z",
                                            "lte": f"{current_date_str}T23:59:59.999Z"
                                        }
                                    }
                                }
                            ],
                            "must_not": {
                                "exists": {
                                    "field": "primary_topics"
                                }
                            }
                        }
                    }
                }
            else:
                logger.info(f"Url: {url}, Start Date: {start_date_str}, Current Date: {current_date_str}")
                query = {
                    "query": {
                        "bool": {
                            "must_not": {
                                "exists": {
                                    "field": "primary_topics"
                                }
                            }
                        }
                    }
                }

            # Initialize the scroll
            scroll_response = self._es_client.search(index=es_index, body=query, size=self._es_data_fetch_size,
                                                     scroll='1m')
            scroll_id = scroll_response['_scroll_id']
            results = scroll_response['hits']['hits']

            # Dump the documents into the json file
            logger.info(f"Starting dumping of {es_index} data in json...")
            while len(results) > 0:
                # Save the current batch of results
                for result in results:
                    output_list.append(result)

                # Fetch the next batch of results
                scroll_response = self._es_client.scroll(scroll_id=scroll_id, scroll='1m')
                scroll_id = scroll_response['_scroll_id']
                results = scroll_response['hits']['hits']

            logger.info(
                f"Dumping of {es_index} data in json has completed and has taken {time.time() - start_time:.2f} seconds.")

            return output_list
        else:
            logger.warning('Could not connect to Elasticsearch')
            return output_list

    @property
    def es_client(self):
        return self._es_client


def empty_dir(file_path):
    try:
        shutil.rmtree(file_path)
    except:
        pass
    finally:
        os.makedirs(file_path, exist_ok=True)


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.
    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True
    except ValueError:
        return False


def normalize_text(s, sep_token=" \n "):
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r". ,", "", s)
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.replace("#", "")
    s = s.strip()
    return s


def preprocess_email(email_body):
    email_body = email_body.split("-------------- next part --------------")[0]
    email_lines = email_body.split('\n')
    temp_ = []
    for line in email_lines:
        if line.startswith("On"):
            line = line.replace("-", " ")
            x = re.sub('\d', ' ', line)
            if is_date(x, fuzzy=True):
                continue
            if line.endswith("> wrote:"):
                continue
        if line.endswith("> wrote:"):
            continue
        if line.startswith("Le "):
            continue
        if line.endswith("?crit :"):
            continue
        if line and not line.startswith('>'):
            if line.startswith('-- ') or line.startswith('[') or line.startswith('_____'):
                continue
            temp_.append(line)
    email_string = "\n".join(temp_)
    normalized_email_string = normalize_text(email_string)
    return normalized_email_string


# pre-compile the patterns
regex_url = re.compile(r'http\S+|www\S+|https\S+')
regex_non_alpha = re.compile(r'[^A-Za-z0-9 ]+')
regex_spaces = re.compile(r'\s+')


def clean_text(text):
    text = regex_url.sub('', text.lower())  # remove urls and convert to lower case
    text = regex_non_alpha.sub('', text)  # remove non-alphanumeric characters
    text = regex_spaces.sub(' ', text).strip()  # replace whitespace sequences with a single space
    return text


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
