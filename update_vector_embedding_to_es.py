import os
from dotenv import load_dotenv
import warnings
import tqdm
import traceback
from loguru import logger
from datetime import datetime, timedelta
from src.utils import log_csv
from src.config import ES_INDEX, EMBEDDING_MODEL
from src.elasticsearch_utils import ElasticSearchClient

warnings.filterwarnings("ignore")
load_dotenv()

if __name__ == "__main__":

    try:
        # logs automatically rotate log file
        os.makedirs("logs", exist_ok=True)
        logger.add(f"logs/update_vector_embedding_to_es.log", rotation="23:59")

        delay = 3
        elastic_search = ElasticSearchClient()

        '''
        # to add the new field to the index - run once only
        elastic_search.add_vector_field(ES_INDEX, "summary_vector_embeddings")
        '''

        dev_urls = [
            "https://lists.linuxfoundation.org/pipermail/lightning-dev/",
            "https://lists.linuxfoundation.org/pipermail/bitcoin-dev/",
            "https://delvingbitcoin.org/",
            "https://gnusha.org/pi/bitcoindev/",
            "https://mailing-list.bitcoindevs.xyz/bitcoindev/",
        ]

        url_status = {}

        for dev_url in dev_urls:
            updated_count = set()
            error_message = "---"
            error_occurred = False
            try:
                logger.info(f"dev_url: {dev_url}")
                dev_name = dev_url.split("/")[-2]

                # if APPLY_DATE_RANGE is set to False, elasticsearch will fetch from all the docs in the index
                APPLY_DATE_RANGE = False

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
                    es_index=ES_INDEX, url=dev_url, field_name="summary_vector_embeddings",
                    start_date_str=start_date_str, current_date_str=current_date_str
                )
                logger.success(f"TOTAL THREADS RECEIVED WITH AN EMPTY 'summary_vector_embeddings': {len(docs_list)}")

                if docs_list:
                    for idx, doc in enumerate(tqdm.tqdm(docs_list)):
                        doc_id = doc['_id']
                        doc_index = doc['_index']

                        doc_text = doc['_source'].get('title')
                        doc_summary = doc['_source'].get('summary')
                        if not doc_summary:
                            doc_text += f" \n{doc['_source'].get('body')}"
                        else:
                            doc_text += f" \n{doc_summary}"

                        if not doc['_source'].get('summary_vector_embeddings') and doc_summary:
                            try:
                                text_vector = EMBEDDING_MODEL.encode(doc_text, normalize_embeddings=True).tolist()

                                if text_vector:
                                    res = elastic_search.es_client.update(
                                        index=doc_index,
                                        id=doc_id,
                                        body={
                                            "doc": {
                                                "summary_vector_embeddings": text_vector
                                            },
                                            "doc_as_upsert": True  # insert the document if it does not already exist
                                        }
                                    )
                                    updated_count.add(res["_id"])
                                else:
                                    logger.info(
                                        f"Nothing to update! Vector length: {len(text_vector)}, '_id': {doc_id}")
                            except Exception as ex:
                                logger.error(f"Error updating ES index: {ex} \n{traceback.format_exc()}")
                        else:
                            if not doc_summary:
                                logger.info(
                                    f"'summary' doesn't exist for '_id': {doc_id} | {doc['_source']['created_at']}")
                logger.success(f"Process complete for dev_url: {dev_url}")

            except Exception as ex:
                error_occurred = True
                error_message = str(ex)

            finally:
                log_csv(file_name="update_vector_embedding_to_es",
                        url=dev_url,
                        updated=len(updated_count),
                        error_log=error_message,
                        error=str(error_occurred)
                        )
                url_status[dev_url] = {
                    "updated_count": len(updated_count),
                    "error_occurred": error_occurred,
                }

        logger.success("Process complete for all domain ")
        logger.info("Summary of updates for all URLs:")
        for url, status in url_status.items():
            logger.info(
                f"URL: {url} - Updated Count: {status['updated_count']} - Error Occurred: {status['error_occurred']}.")

    except Exception as ex:
        logger.error(f"Error: {str(ex)}\n{traceback.format_exc()}")
