from loguru import logger
import os
from dotenv import load_dotenv
import warnings
import tqdm
import pandas as pd
import traceback

from src.config import ES_CLOUD_ID, ES_USERNAME, ES_PASSWORD, ES_INDEX
from src.utils import ElasticSearchClient

warnings.filterwarnings("ignore")
load_dotenv()

# logs automatically rotate log file
os.makedirs("logs1", exist_ok=True)
logger.add(f"logs1/generate_topics_modeling.log", rotation="23:59")

if __name__ == "__main__":

    btc_topics_list = pd.read_csv("btc_topics.csv")
    btc_topics_list = btc_topics_list['Topics'].to_list()

    elastic_search = ElasticSearchClient(es_cloud_id=ES_CLOUD_ID, es_username=ES_USERNAME,
                                         es_password=ES_PASSWORD)

    dev_urls = [
        "https://lists.linuxfoundation.org/pipermail/lightning-dev/",
        "https://lists.linuxfoundation.org/pipermail/bitcoin-dev/",
    ]

    for dev_url in dev_urls:
        dev_name = dev_url.split("/")[-2]
        logger.info(f"dev_url: {dev_url}")
        logger.info(f"dev_name: {dev_name}")

        elastic_search = ElasticSearchClient(es_cloud_id=ES_CLOUD_ID, es_username=ES_USERNAME,
                                             es_password=ES_PASSWORD)

        for topic in btc_topics_list:
            # fetch all docs that matches the provided topic
            logger.info(f"Fetching docs for topic: {topic}")
            docs_list = elastic_search.fetch_docs_with_keywords(ES_INDEX, dev_url, topic)
            logger.success(f"TOTAL THREADS RECEIVED WITH A TOPIC {str(topic).upper()}: {len(docs_list)}")

            if docs_list:
                # update topic to each doc
                logger.info(f'updating topic to all these docs...')
                for idx, doc in enumerate(tqdm.tqdm(docs_list)):
                    try:

                        doc_source_id = doc['_source']['id']
                        doc_id = doc['_id']
                        doc_index = doc['_index']
                        logger.info(f"Doc Id: {doc_id}, Source Id: {doc_source_id}")

                        # update primary keyword
                        primary_kw = doc['_source'].get('primary_topics')
                        if primary_kw:
                            primary_kw.append(topic)
                            primary_kw = list(set(primary_kw))

                        elastic_search.es_client.update(
                            index=doc_index,
                            id=doc_id,
                            body={
                                'doc': {
                                    "primary_topics": primary_kw if primary_kw else []
                                }
                            }
                        )

                    except Exception as ex:
                        logger.error(f"Error occurred while updating topics: {ex} \n{traceback.format_exc()}")

            else:
                logger.info(f"NO THREADS FOUND FOR A TOPIC: {str(topic).upper()}")

            logger.info(f"Process completed for dev_url: {dev_url}")
