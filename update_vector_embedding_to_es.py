import os
from dotenv import load_dotenv
import warnings
import tqdm
import traceback
from loguru import logger
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

from src.config import ES_CLOUD_ID, ES_USERNAME, ES_PASSWORD, ES_INDEX
from src.utils import ElasticSearchClient

warnings.filterwarnings("ignore")
load_dotenv()

# logs automatically rotate log file
os.makedirs("logs", exist_ok=True)
logger.add(f"logs/update_vector_embedding_to_es.log", rotation="23:59")

# define embedding model
model = SentenceTransformer('intfloat/e5-large-v2')  # intfloat/e5-base-v2

if __name__ == "__main__":

    delay = 3
    elastic_search = ElasticSearchClient(es_cloud_id=ES_CLOUD_ID, es_username=ES_USERNAME,
                                         es_password=ES_PASSWORD)

    '''
    # to add the new field to the index - run once only
    elastic_search.add_vector_field(ES_INDEX, "summary_vector")
    '''

    dev_urls = [
        "https://lists.linuxfoundation.org/pipermail/lightning-dev/",
        "https://lists.linuxfoundation.org/pipermail/bitcoin-dev/",
        "https://delvingbitcoin.org/"
    ]

    for dev_url in dev_urls:
        logger.info(f"dev_url: {dev_url}")
        dev_name = dev_url.split("/")[-2]

        # if APPLY_DATE_RANGE is set to False, elasticsearch will fetch all the docs in the index
        APPLY_DATE_RANGE = True

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

        docs_list = elastic_search.fetch_data_for_empty_vector_embedding(ES_INDEX, dev_url, start_date_str,
                                                                         current_date_str)
        logger.success(f"TOTAL THREADS RECEIVED WITH AN EMPTY VECTORS: {len(docs_list)}")

        if docs_list:
            for idx, doc in enumerate(tqdm.tqdm(docs_list)):
                doc_source_id = doc['_source']['id']
                doc_id = doc['_id']
                doc_index = doc['_index']
                logger.info(f"Doc Id: {doc_id}, Source Id: {doc_source_id}")

                doc_text = doc['_source'].get('title')
                doc_summary = doc['_source'].get('summary')
                if not doc_summary:
                    doc_text += f" \n{doc['_source'].get('body')}"
                else:
                    doc_text += f" \n{doc_summary}"

                if not doc['_source'].get('summary_vector'):
                    try:
                        text_vector = model.encode(doc_text, normalize_embeddings=True)

                        if text_vector.any():
                            res = elastic_search.es_client.update(
                                index=doc_index,
                                id=doc_id,
                                body={
                                    'doc': {
                                        "summary_vector": text_vector
                                    }
                                }
                            )
                            logger.info(res)
                            break

                        else:
                            logger.info(
                                f"Nothing to update! Vector length: {len(text_vector)} Doc Id: {doc_id}, Source Id: {doc_source_id}")

                    except Exception as ex:
                        logger.error(f"Error updating ES index: {ex} \n{traceback.format_exc()}")

        logger.success(f"Process complete for dev_url: {dev_url}")
