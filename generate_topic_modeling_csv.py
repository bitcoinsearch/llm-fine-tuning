from datetime import datetime, timedelta
from loguru import logger
import os
from dotenv import load_dotenv
import warnings
import tqdm
import pandas as pd
import traceback
import time

from src.config import ES_INDEX
from src.utils import preprocess_email
from src.elasticsearch_utils import ElasticSearchClient
from src.gpt_utils import apply_topic_modeling

warnings.filterwarnings("ignore")
load_dotenv()


if __name__ == "__main__":

    # logs automatically rotate log file
    os.makedirs("logs", exist_ok=True)
    logger.add(f"logs/generate_topics_modeling.log", rotation="23:59")

    delay = 3
    btc_topics_list = pd.read_csv("btc_topics.csv")
    btc_topics_list = btc_topics_list['Topics'].to_list()

    elastic_search = ElasticSearchClient()

    dev_urls = [
        "https://lists.linuxfoundation.org/pipermail/lightning-dev/",
        "https://lists.linuxfoundation.org/pipermail/bitcoin-dev/",
        "https://delvingbitcoin.org/",
        "https://gnusha.org/pi/bitcoindev/",
        # "all_data", # uncomment this line if you want to generate topic modeling on all docs
    ]

    for dev_url in dev_urls:

        if dev_url == "all_data":
            dev_name = "all_data"
            dev_url = None
        else:
            dev_name = dev_url.split("/")[-2]

        logger.info(f"dev_url: {dev_url}")
        logger.info(f"dev_name: {dev_name}")

        # if APPLY_DATE_RANGE is set to False, elasticsearch will fetch all the docs in the index
        APPLY_DATE_RANGE = False

        # if UPDATE_ES_SIMULTANEOUSLY set to True, it will update topics in the elasticsearch docs as we generate them
        UPDATE_ES_SIMULTANEOUSLY = False

        # if SAVE_CSV is set to True, it will store generated topics data into csv file
        SAVE_CSV = True
        SAVE_AT_MULTIPLE_OF = 50

        OUTPUT_DIR = "gpt_output"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        CSV_FILE_PATH = f"{OUTPUT_DIR}/topic_modeling_{dev_name}.csv"

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

        docs_list = elastic_search.fetch_data_for_empty_field(
            es_index=ES_INDEX, url=dev_url, field_name="primary_topics",
            start_date_str=start_date_str, current_date_str=current_date_str
        )
        logger.success(f"TOTAL THREADS RECEIVED WITH AN EMPTY FIELD - 'primary_topics': {len(docs_list)}")

        if docs_list:
            if os.path.exists(CSV_FILE_PATH):
                stored_df = pd.read_csv(CSV_FILE_PATH)
                logger.info(f"Shape of stored df: {stored_df.shape}")

                stored_source_ids = stored_df['source_id'].to_list()
                logger.info(f"Docs in stored df: {len(stored_source_ids)}")
            else:
                logger.info(f"CSV file path does not exist! Creating new one: {CSV_FILE_PATH}")
                stored_df = pd.DataFrame(columns=['primary_topics', 'secondary_topics', 'source_id'])
                stored_source_ids = stored_df['source_id'].to_list()

            for idx, doc in enumerate(tqdm.tqdm(docs_list)):
                doc_source_id = doc['_source']['id']

                if CSV_FILE_PATH:
                    if doc_source_id in stored_source_ids:
                        continue

                doc_id = doc['_id']
                doc_index = doc['_index']
                logger.info(f"_id: {doc_id} | title: {doc['_source']['title']}")

                doc_body = doc['_source'].get('summary', '')
                if not doc_body:
                    doc_body = doc['_source'].get('body', '')
                    doc_body = preprocess_email(email_body=doc_body)

                if not doc['_source'].get('primary_topics'):
                    doc_text = ""
                    if doc_body:
                        doc_title = doc['_source'].get('title')
                        doc_text = doc_title + "\n" + doc_body

                    if doc_text:
                        primary_kw, secondary_kw = [], []
                        try:
                            primary_kw, secondary_kw = apply_topic_modeling(text=doc_text, topic_list=btc_topics_list)

                            if SAVE_CSV and not UPDATE_ES_SIMULTANEOUSLY:
                                row_data = {
                                    'primary_topics': primary_kw if primary_kw else [],
                                    'secondary_topics': secondary_kw if secondary_kw else [],
                                    'source_id': doc_source_id if doc_source_id else None
                                }
                                row_data = pd.Series(row_data).to_frame().T
                                stored_df = pd.concat([stored_df, row_data], ignore_index=True)

                                if idx % SAVE_AT_MULTIPLE_OF == 0:
                                    stored_df.drop_duplicates(subset='source_id', keep='first', inplace=True)
                                    stored_df.to_csv(CSV_FILE_PATH, index=False)
                                    time.sleep(delay)
                                    logger.info(f"csv file saved at IDX: {idx}, PATH: {CSV_FILE_PATH}")

                            elif UPDATE_ES_SIMULTANEOUSLY and not SAVE_CSV:
                                # update primary keyword
                                elastic_search.es_client.update(
                                    index=doc_index,
                                    id=doc_id,
                                    body={
                                        'doc': {
                                            "primary_topics": primary_kw if primary_kw else []
                                        }
                                    }
                                )
                                # update secondary keyword
                                elastic_search.es_client.update(
                                    index=doc_index,
                                    id=doc_id,
                                    body={
                                        'doc': {
                                            "secondary_topics": secondary_kw if secondary_kw else []
                                        }
                                    }
                                )

                            elif SAVE_CSV and UPDATE_ES_SIMULTANEOUSLY:
                                # update primary keyword
                                elastic_search.es_client.update(
                                    index=doc_index,
                                    id=doc_id,
                                    body={
                                        'doc': {
                                            "primary_topics": primary_kw if primary_kw else []
                                        }
                                    }
                                )
                                # update secondary keyword
                                elastic_search.es_client.update(
                                    index=doc_index,
                                    id=doc_id,
                                    body={
                                        'doc': {
                                            "secondary_topics": secondary_kw if secondary_kw else []
                                        }
                                    }
                                )

                                # store in csv file
                                row_data = {
                                    'primary_topics': primary_kw if primary_kw else [],
                                    'secondary_topics': secondary_kw if secondary_kw else [],
                                    'source_id': doc_source_id if doc_source_id else None
                                }
                                row_data = pd.Series(row_data).to_frame().T
                                stored_df = pd.concat([stored_df, row_data], ignore_index=True)

                                if idx % SAVE_AT_MULTIPLE_OF == 0:
                                    stored_df.drop_duplicates(subset='source_id', keep='first', inplace=True)
                                    stored_df.to_csv(CSV_FILE_PATH, index=False)
                                    time.sleep(delay)
                                    logger.info(f"csv file saved at IDX: {idx}, PATH: {CSV_FILE_PATH}")

                            else:  # not SAVE_CSV and not UPDATE_ES_SIMULTANEOUSLY
                                pass

                        except Exception as ex:
                            logger.error(f"Error: apply_topic_modeling: {str(ex)}\n{traceback.format_exc()}")

                            stored_df.drop_duplicates(subset='source_id', keep='first', inplace=True)
                            stored_df.to_csv(CSV_FILE_PATH, index=False)
                            time.sleep(delay)
                            logger.info(f"csv file saved at IDX: {idx}, PATH: {CSV_FILE_PATH}")

                    else:
                        logger.warning(f"Body Text not found! Doc ID: {doc_id}")

            stored_df.drop_duplicates(subset='source_id', keep='first', inplace=True)
            stored_df.to_csv(CSV_FILE_PATH, index=False)
            time.sleep(delay)
            logger.success(f"FINAL CSV FILE SAVED AT PATH: {CSV_FILE_PATH}")

        logger.info(f"Process completed for dev_url: {dev_url}")
