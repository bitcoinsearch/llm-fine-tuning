import re
from elasticsearch import Elasticsearch
import time
import openai
from datetime import datetime, timedelta
from loguru import logger
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv
import warnings
import pytz
import tqdm
import pandas as pd

from src.utils import preprocess_email
from src.config import TOKENIZER, ES_CLOUD_ID, ES_USERNAME, ES_PASSWORD, ES_INDEX, ES_DATA_FETCH_SIZE

warnings.filterwarnings("ignore")
load_dotenv()

# if set to True, it will use chatgpt model ("gpt-3.5-turbo") for all the completions
CHATGPT = True

# COMPLETION_MODEL - only applicable if CHATGPT is set to False
OPENAI_ORG_KEY = os.getenv("OPENAI_ORG_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.organization = OPENAI_ORG_KEY
openai.api_key = OPENAI_API_KEY


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
                scroll='1m'
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
                scroll_response = self._es_client.scroll(scroll_id=scroll_id, scroll='1m')
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


if __name__ == "__main__":

    # if production is set to False, elasticsearch will fetch all the docs in the index
    PRODUCTION = True

    # workflow
    xml_reader = XMLReader()
    elastic_search = ElasticSearchClient(es_cloud_id=ES_CLOUD_ID, es_username=ES_USERNAME,
                                         es_password=ES_PASSWORD)

    dev_urls = [
        "https://lists.linuxfoundation.org/pipermail/bitcoin-dev/",
        "https://lists.linuxfoundation.org/pipermail/lightning-dev/"
    ]

    for dev_url in dev_urls:
        logger.info(f"dev_url: {dev_url}")

        if PRODUCTION:
            current_date_str = None
            if not current_date_str:
                current_date_str = datetime.now().strftime("%Y-%m-%d")
            start_date = datetime.now() - timedelta(days=7)
            start_date_str = start_date.strftime("%Y-%m-%d")
            logger.info(f"start_date: {start_date_str}")
            logger.info(f"current_date_str: {current_date_str}")

            docs_list = elastic_search.extract_data_from_es(ES_INDEX, dev_url, start_date_str, current_date_str)

        else:
            docs_list = elastic_search.fetch_all_data_for_url(ES_INDEX, dev_url)

        dev_name = dev_url.split("/")[-2]
        logger.success(f"Total threads received for {dev_name}: {len(docs_list)}")

        # variables
        # docs_list = docs_list[:100]  # delete this line after testing
        CSV_FILE_PATH = "data.csv"

        if os.path.exists(CSV_FILE_PATH):
            os.remove(CSV_FILE_PATH)

        dataset = []

        for doc in tqdm.tqdm(docs_list):
            res = None
            try:
                email_body = doc['_source'].get('body')
                email_summary = doc['_source'].get('summary')

                if email_body and email_summary:
                    preprocessed_email_body = preprocess_email(email_body=email_body)

                    B_INST, E_INST = "[INST]", "[/INST]"
                    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                    assistant_instruct = "You are an intelligent assistant."

                    template = f"""<s>{B_SYS}{assistant_instruct}{E_SYS}</s> {B_INST} ### Input: Suppose you are a programmer and you are enriched by programming knowledge. You will be going through other programmers mail sent to you and you will be extracting all the important information out of the mail and composing a blog post. Even if the mail is divided into parts and parts, your extraction summary should not be in bullet points. It should be in multiple paragraphs. I repeat, never in bullet points. You have to follow some rules while giving a detailed summary. 
                    The rules are below:
                        1. While extracting, avoid using phrases referring to the context. Instead, directly present the information or points covered.  Do not introduce sentences with phrases like: "The context discusses...", "In this context..." or "The context covers..." or "The context questions..." etc
                        2. The summary tone should be formal and full of information.
                        3. Add spaces after using punctuation and follow all the grammatical rules.
                        4. Try to retain all the links provided and use them in proper manner at proper place.
                        5. The farewell part of the email should be completely ignored.
                        6. Ensure that summary is not simply a rephrase of the original content with minor word changes, but a restructured and simplified rendition of the main points.
                        7. Most importantly, this extracted information should be relative of the size of the email. If it is a bigger email, the extracted summary can be longer than a very short email. 
                        Context: {preprocessed_email_body}\n

                        ### Output: {E_INST} {email_summary}"""

                    dataset.append({"text": template})

                else:
                    logger.warning(f"Email body: {bool(email_body)}, Email Summary: {bool(email_summary)}")

            except Exception as ex:
                error_message = f"Error occurred: {ex}"
                if res:
                    error_message += f", Response: {res}"
                logger.error(error_message)

        df = pd.DataFrame(dataset)
        df.to_csv(CSV_FILE_PATH, index=False)
        logger.success(f"CSV file generated successfully: {CSV_FILE_PATH}")
