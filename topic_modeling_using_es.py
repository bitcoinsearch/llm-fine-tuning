from loguru import logger
import os
from dotenv import load_dotenv
import warnings
import tqdm
import pandas as pd
import traceback
import time

from src.config import ES_CLOUD_ID, ES_USERNAME, ES_PASSWORD, ES_INDEX
from src.utils import ElasticSearchClient, clean_title

warnings.filterwarnings("ignore")
load_dotenv()

# logs automatically rotate log file
os.makedirs("es_topic_modeling_logs", exist_ok=True)
logger.add(f"es_topic_modeling_logs/generate_topics_modeling.log", rotation="23:59")

if __name__ == "__main__":

    btc_topics_list = pd.read_csv("btc_topics.csv")
    btc_topics_list = btc_topics_list['Topics'].to_list()

    elastic_search = ElasticSearchClient(es_cloud_id=ES_CLOUD_ID, es_username=ES_USERNAME,
                                         es_password=ES_PASSWORD)

    dev_urls = [
        "https://lists.linuxfoundation.org/pipermail/lightning-dev/",
        "https://lists.linuxfoundation.org/pipermail/bitcoin-dev/",
        "https://delvingbitcoin.org/"
    ]

    for dev_url in dev_urls:
        dev_name = dev_url.split("/")[-2]
        logger.info(f"dev_url: {dev_url}")
        logger.info(f"dev_name: {dev_name}")

        elastic_search = ElasticSearchClient(es_cloud_id=ES_CLOUD_ID, es_username=ES_USERNAME,
                                             es_password=ES_PASSWORD)

        for topic in btc_topics_list:

            SAVE_CSV = True
            OUTPUT_DIR = "es_topic_modeling_output"
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            CSV_FILE_PATH = fr"{OUTPUT_DIR}/{clean_title(topic)}.csv"

            # fetch all docs that matches the provided topic
            logger.info(f"Fetching docs for topic: {topic}")
            docs_list = elastic_search.fetch_docs_with_keywords(ES_INDEX, dev_url, topic)
            logger.success(f"TOTAL THREADS RECEIVED WITH A TOPIC: {str(topic)} = {len(docs_list)}")

            if docs_list:
                if os.path.exists(CSV_FILE_PATH):
                    stored_df = pd.read_csv(CSV_FILE_PATH)
                    logger.info(f"Shape of stored df: {stored_df.shape}")
                    stored_source_ids = stored_df['source_id'].to_list()
                    logger.info(f"Docs in stored df: {len(stored_source_ids)}")
                else:
                    logger.info(f"CSV file path does not exist! Creating new one: {CSV_FILE_PATH}")
                    stored_df = pd.DataFrame(columns=['primary_topics', 'source_id', 'source_url'])
                    stored_source_ids = stored_df['source_id'].to_list()

                # update topic to each doc
                logger.info(f'updating topic to all these docs...')

                for idx, doc in enumerate(tqdm.tqdm(docs_list)):
                    try:
                        doc_source_id = doc['_source']['id']
                        doc_source_url = doc['_source']['url']
                        doc_id = doc['_id']
                        doc_index = doc['_index']
                        logger.info(f"Doc Id: {doc_id}, Source Id: {doc_source_id}")

                        # update primary keyword
                        primary_kw = doc['_source'].get('primary_topics', [])
                        primary_kw.append(topic)
                        primary_kw = list(set(primary_kw))

                        # update topics to elasticsearch
                        elastic_search.es_client.update(
                            index=doc_index,
                            id=doc_id,
                            body={
                                'doc': {
                                    "primary_topics": primary_kw if primary_kw else []
                                }
                            }
                        )

                        # save csv for each topic with top 5 docs and their topics
                        if idx <= 5 and SAVE_CSV:
                            row_data = {
                                'primary_topics': primary_kw if primary_kw else [],
                                'source_id': doc_source_id if doc_source_id else None,
                                'source_url': doc_source_url if doc_source_url else None
                            }
                            row_data = pd.Series(row_data).to_frame().T
                            stored_df = pd.concat([stored_df, row_data], ignore_index=True)
                            stored_df.drop_duplicates(subset='source_id', keep='first', inplace=True)
                            stored_df.to_csv(CSV_FILE_PATH, index=False)
                            time.sleep(2)
                            logger.info(f"csv file saved at IDX: {idx}, PATH: {CSV_FILE_PATH}")

                    except Exception as ex:
                        logger.error(f"Error occurred while updating topics: {ex} \n{traceback.format_exc()}")
            else:
                logger.info(f"NO THREADS FOUND FOR A TOPIC: {str(topic).upper()}")

            logger.info(f"Process completed for dev_url: {dev_url}")
