import os
import re
import shutil
import traceback
import pandas as pd
from dotenv import load_dotenv
from dateutil.parser import parse
from elasticsearch import Elasticsearch
from datetime import datetime
import time
from loguru import logger
import warnings
import tiktoken
import pytz
import xml.etree.ElementTree as ET

from src.config import TOKENIZER, ES_DATA_FETCH_SIZE

warnings.filterwarnings("ignore")
load_dotenv()


def clean_title(xml_name):
    special_characters = ['/', ':', '@', '#', '$', '*', '&', '<', '>', '\\', '?']
    xml_name = re.sub(r'[^A-Za-z0-9]+', '-', xml_name)
    for sc in special_characters:
        xml_name = xml_name.replace(sc, "-")
    return xml_name


class XMLReader:
    def __init__(self) -> None:
        self.month_dict = {
            1: "Jan", 2: "Feb", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"
        }

    def get_id(self, id):
        return str(id).split("-")[-1]

    def clean_title(self, xml_name):
        special_characters = ['/', ':', '@', '#', '$', '*', '&', '<', '>', '\\', '?']
        xml_name = re.sub(r'[^A-Za-z0-9]+', '-', xml_name)
        for sc in special_characters:
            xml_name = xml_name.replace(sc, "-")
        return xml_name

    def get_xml_summary(self, data, dev_name):
        number = self.get_id(data["_source"]["id"])
        title = data["_source"]["title"]
        xml_name = self.clean_title(title)
        published_at = datetime.strptime(data['_source']['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')
        published_at = pytz.UTC.localize(published_at)
        month_name = self.month_dict[int(published_at.month)]
        str_month_year = f"{month_name}_{int(published_at.year)}"
        current_directory = os.getcwd()
        file_path = f"static/{dev_name}/{str_month_year}/{number}_{xml_name}.xml"
        full_path = os.path.join(current_directory, file_path)

        try:
            if os.path.exists(full_path):
                namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
                tree = ET.parse(full_path)
                root = tree.getroot()
                summ_list = root.findall(".//atom:entry/atom:summary", namespaces)
                if summ_list:
                    summ = "\n".join([summ.text for summ in summ_list])
                    return summ, f"Summary text found: {full_path}"
                else:
                    return None, f"No summary found: {full_path}"
            else:
                return None, f"No xml file found: {full_path}"
        except Exception as e:
            ex_message = f"Error: {e} at file path: {full_path}"
            logger.error(ex_message)
            return None, ex_message


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
            # not url and not start_date_str and not current_date_str
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
                                                     scroll='5m')
            scroll_id = scroll_response['_scroll_id']
            results = scroll_response['hits']['hits']

            # Dump the documents into the json file
            logger.info(f"Starting dumping of {es_index} data in json...")
            while len(results) > 0:
                # Save the current batch of results
                for result in results:
                    output_list.append(result)

                # Fetch the next batch of results
                scroll_response = self._es_client.scroll(scroll_id=scroll_id, scroll='5m')
                scroll_id = scroll_response['_scroll_id']
                results = scroll_response['hits']['hits']

            logger.info(
                f"Dumping of {es_index} data in json has completed and has taken {time.time() - start_time:.2f} seconds.")

            return output_list
        else:
            logger.warning('Could not connect to Elasticsearch')
            return output_list

    def fetch_all_data_for_url(self, es_index, url):
        logger.info(f"fetching all the data")
        output_list = []
        start_time = time.time()

        if self._es_client.ping():
            logger.info("connected to the ElasticSearch")
            query = {
                "query": {
                    "match_phrase": {
                        "domain": str(url)
                    }
                }
            }

            # Initialize the scroll
            scroll_response = self._es_client.search(index=es_index, body=query, size=self._es_data_fetch_size,
                                                     scroll='5m')
            scroll_id = scroll_response['_scroll_id']
            results = scroll_response['hits']['hits']

            # Dump the documents into the json file
            logger.info(f"Starting dumping of {es_index} data in json...")
            while len(results) > 0:
                # Save the current batch of results
                for result in results:
                    output_list.append(result)

                # Fetch the next batch of results
                scroll_response = self._es_client.scroll(scroll_id=scroll_id, scroll='5m')
                scroll_id = scroll_response['_scroll_id']
                results = scroll_response['hits']['hits']

            logger.info(
                f"Dumping of {es_index} data in json has completed and has taken {time.time() - start_time:.2f} seconds.")

            return output_list
        else:
            logger.info('Could not connect to Elasticsearch')
            return None

    def extract_data_from_es(self, es_index, url, start_date_str, current_date_str):
        output_list = []
        start_time = time.time()

        if self._es_client.ping():
            logger.success("connected to the ElasticSearch")
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "prefix": {  # Using prefix query for domain matching
                                    "domain.keyword": str(url)
                                }
                            },
                            {
                                "range": {
                                    "created_at": {
                                        "gte": f"{start_date_str}T00:00:00.000Z",
                                        "lte": f"{current_date_str}T23:59:59.999Z"
                                    }
                                }
                            }
                        ]
                    }
                }
            }

            # Initialize the scroll
            scroll_response = self._es_client.search(
                index=es_index,
                body=query,
                size=self._es_data_fetch_size,
                scroll='5m'
            )
            scroll_id = scroll_response['_scroll_id']
            results = scroll_response['hits']['hits']

            # Dump the documents into the json file
            logger.info(f"Starting dumping of {es_index} data in json...")
            while len(results) > 0:
                # Save the current batch of results
                for result in results:
                    output_list.append(result)

                # Fetch the next batch of results
                scroll_response = self._es_client.scroll(scroll_id=scroll_id, scroll='5m')
                scroll_id = scroll_response['_scroll_id']
                results = scroll_response['hits']['hits']

            logger.info(
                f"Dumping of {es_index} data in json has completed and has taken {time.time() - start_time:.2f} seconds.")

            return output_list
        else:
            logger.info('Could not connect to Elasticsearch')
            return None

    def fetch_docs_with_keywords(self, es_index, url, keyword):
        output_list = []
        start_time = time.time()

        if self._es_client.ping():
            logger.success("connected to the ElasticSearch")

            query = {
                "min_score": 1,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "prefix": {
                                    "domain.keyword": str(url)
                                }
                            },
                            {
                                "match_phrase": {
                                    "summary": str(keyword)
                                }
                            },
                            {
                                "match_phrase": {
                                    "body": str(keyword)
                                }
                            },
                            {
                                "match": {
                                    "summary": {
                                        "query": str(keyword),
                                        "minimum_should_match": "95%",
                                        # "operator": "and"
                                    }
                                }
                            },
                            {
                                "match": {
                                    "body": {
                                        "query": str(keyword),
                                        "minimum_should_match": "95%",
                                        # "operator": "and"
                                    }
                                }
                            }
                        ]
                    }
                }
            }

            # Initialize the scroll
            scroll_response = self._es_client.search(
                index=es_index,
                body=query,
                size=self._es_data_fetch_size,
                scroll='5m'
            )
            scroll_id = scroll_response['_scroll_id']
            results = scroll_response['hits']['hits']

            # Dump the documents into the json file
            logger.info(f"Starting dumping of {es_index} data in json...")
            while len(results) > 0:
                # Save the current batch of results
                for result in results:
                    output_list.append(result)

                # Fetch the next batch of results
                scroll_response = self._es_client.scroll(scroll_id=scroll_id, scroll='5m')
                scroll_id = scroll_response['_scroll_id']
                results = scroll_response['hits']['hits']

            logger.info(
                f"Dumping of {es_index} data in json has completed and has taken {time.time() - start_time:.2f} seconds.")

            return output_list
        else:
            logger.info('Could not connect to Elasticsearch')
            return None

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


def merge_multiple_csv_from_dir(dir_path, csv_save_path):
    # get a list of all the csv files
    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    logger.info(f"Total csv files found: {len(csv_files)}")

    # initialize a list to store dataframes
    dfs = []

    # loop through the list of csv files
    for file in csv_files:
        try:
            # read the csv file
            curr_df = pd.read_csv(dir_path + file)
            logger.info(f"individual file shape bfr:{curr_df.shape}")

            # drop duplicate rows based on the 'source_id' column
            curr_df.drop_duplicates(subset='source_id', keep='first', inplace=True)
            logger.info(f"individual file shape aft:{curr_df.shape}")

            # append the current df to the list
            dfs.append(curr_df)
        except Exception as ex:
            logger.error(f"Error Occurred: {ex}")
            logger.error(traceback.format_exc())

    # concatenate all dataframes in the list
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"main file shape bfr:{df.shape}")

    # drop duplicates from the final dataframe
    df.drop_duplicates(subset='source_id', keep='first', inplace=True)
    logger.info(f"main file shape aft:{df.shape}")

    # write the final dataframe to a csv file
    df.to_csv(f"{csv_save_path}", index=False)
    logger.success(f"file saved at path: {csv_save_path}")


def convert_to_dataframe(docs_list, save_csv=False):
    dict_list = []
    for i in docs_list:
        src = i.get("_source")
        i_data = {
            "index": str(i.get("_index", "")).strip(),
            "_id": str(i.get("_id", "")).strip(),
            "title": str(src.get("title", "")).strip(),
            "transcript_by": str(src.get("transcript_by", "")).strip(),
            "domain": str(src.get("domain", "")).strip(),
            "created_at": str(src.get("created_at", "")).strip(),
            "body_type": str(src.get("body_type", "")).strip(),
            "url": str(src.get("url", "")).strip()
        }
        dict_list.append(i_data)

    df = pd.DataFrame(dict_list)
    if save_csv:
        df.to_csv("data_list.csv", index=False)
    return df


def get_duplicated_docs_ids(df):
    logger.info(f"Shape: {df.shape}")
    cols = ['index', 'title', 'transcript_by', 'created_at', 'domain', 'body_type']
    df_grouped = df.groupby(cols)

    ids_to_keep = []
    for _, dfx in df_grouped:

        # if only one instance keep it
        if dfx.shape[0] == 1:
            this_id = dfx['_id'].values[0]
            ids_to_keep.append(this_id)

        else:
            urls_ = list(set(dfx['url'].to_list()))

            # check if all urls are same, if yes then keep the last one
            if len(urls_) == 1:
                this_id = dfx['_id'].values[-1]
                ids_to_keep.append(this_id)

            # if all urls are different
            else:
                # check if last route of each urls are same, if yes move forward with this operation
                last_route_urls = list(set([i.split("/")[-1] for i in urls_]))

                if len(last_route_urls) == 1:
                    base_url = "https://btctranscripts.com/bitcoin-core-dev-tech/"
                    pattern = re.compile("^https://btctranscripts.com/bitcoin-core-dev-tech/[0-9]{4}-[0-9]{2}/.*")

                    temp_urls = []
                    for _, r in dfx.iterrows():
                        this_id = r['_id']
                        this_url = r['url']

                        if this_url not in temp_urls:
                            if base_url in this_url:
                                if pattern.match(this_url):
                                    ids_to_keep.append(this_id)
                            else:
                                ids_to_keep.append(this_id)

                            temp_urls.append(this_url)

                # if last route of each urls are different, take all of them
                else:
                    temp_urls = []
                    for _, r in dfx.iterrows():
                        this_id = r['_id']
                        this_url = r['url']
                        if this_url not in temp_urls:
                            ids_to_keep.append(this_id)
                            temp_urls.append(this_url)

    total_ids = set(df['_id'])
    ids_to_keep_set = set(ids_to_keep)
    ids_to_drop = list(total_ids - ids_to_keep_set)

    df_to_keep = df[df['_id'].isin(ids_to_keep_set)]
    df_to_drop = df[df['_id'].isin(ids_to_drop)]

    duplicates = df_to_keep[df_to_keep.duplicated('url', keep=False)]
    df_to_drop = pd.concat([df_to_drop, duplicates])
    df_to_keep.drop(duplicates.index, inplace=True)

    ids_to_drop = df_to_drop['_id'].to_list()
    logger.info(f"Total: {len(total_ids)}, Keeping: {df_to_keep.shape[0]}, Dropping: {len(ids_to_drop)}")

    return ids_to_drop




