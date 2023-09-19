from datetime import datetime, timedelta
from loguru import logger
import os
from dotenv import load_dotenv
import warnings
import tqdm
import pandas as pd
import traceback
import ast

from src.config import ES_CLOUD_ID, ES_USERNAME, ES_PASSWORD, ES_INDEX
from src.utils import ElasticSearchClient

warnings.filterwarnings("ignore")
load_dotenv()

# logs automatically rotate log file
os.makedirs("logs", exist_ok=True)
logger.add(f"logs/push_topic_modeling_to_es.log", rotation="23:59")


if __name__ == "__main__":

    delay = 3
    elastic_search = ElasticSearchClient(es_cloud_id=ES_CLOUD_ID, es_username=ES_USERNAME,
                                         es_password=ES_PASSWORD)

    dev_urls = [
        "https://lists.linuxfoundation.org/pipermail/lightning-dev/",
        "https://lists.linuxfoundation.org/pipermail/bitcoin-dev/"
    ]

    for dev_url in dev_urls:
        logger.info(f"dev_url: {dev_url}")
        dev_name = dev_url.split("/")[-2]

        # if APPLY_DATE_RANGE is set to False, elasticsearch will fetch all the docs in the index
        APPLY_DATE_RANGE = False

        OUTPUT_DIR = "gpt_output"
        CSV_FILE_PATH = f"{OUTPUT_DIR}/topic_modeling_{dev_name}.csv"

        if os.path.exists(CSV_FILE_PATH):
            stored_df = pd.read_csv(CSV_FILE_PATH)
            logger.info(f"Shape of stored df: {stored_df.shape}")
            stored_df.set_index("source_id", inplace=True)
        else:
            logger.info(f"No data found in CSV! Path: {CSV_FILE_PATH}")
            continue

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
            for idx, doc in enumerate(tqdm.tqdm(docs_list)):
                doc_source_id = doc['_source']['id']
                doc_id = doc['_id']
                doc_index = doc['_index']
                logger.info(f"Doc Id: {doc_id}, Source Id: {doc_source_id}")

                if not doc['_source'].get('primary_topics'):
                    try:
                        this_row = stored_df.loc[doc_source_id]

                        if not this_row.empty:
                            primary_kw = ast.literal_eval(this_row['primary_topics'])
                            secondary_kw = ast.literal_eval(this_row['secondary_topics'])

                            # update primary topic
                            elastic_search.es_client.update(
                                index=doc_index,
                                id=doc_id,
                                body={
                                    'doc': {
                                        "primary_topics": primary_kw if primary_kw else []
                                    }
                                }
                            )

                            # update secondary topic
                            elastic_search.es_client.update(
                                index=doc_index,
                                id=doc_id,
                                body={
                                    'doc': {
                                        "secondary_topics": secondary_kw if secondary_kw else []
                                    }
                                }
                            )
                        else:
                            logger.info(f"No data found for this doc in csv! Doc Id: {doc_id}, Source Id: {doc_source_id}")

                    except Exception as ex:
                        logger.error(f"Error updating ES index: {traceback.format_exc()}")

        logger.success(f"Process complete for dev_url: {dev_url}")
