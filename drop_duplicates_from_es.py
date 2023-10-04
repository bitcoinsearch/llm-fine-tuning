from loguru import logger
from dotenv import load_dotenv
import warnings
import tqdm
import traceback

from src.config import ES_CLOUD_ID, ES_USERNAME, ES_PASSWORD, ES_INDEX
from src.utils import ElasticSearchClient, convert_to_dataframe, get_duplicated_docs_ids

warnings.filterwarnings("ignore")
load_dotenv()


if __name__ == "__main__":

    elastic_search = ElasticSearchClient(es_cloud_id=ES_CLOUD_ID, es_username=ES_USERNAME,
                                         es_password=ES_PASSWORD)

    dev_urls = [
        "https://btctranscripts.com/"
    ]

    for dev_url in dev_urls:

        docs_list = elastic_search.fetch_all_data_for_url(ES_INDEX, dev_url)
        logger.success(f"TOTAL THREADS RECEIVED FOR {dev_url}: {len(docs_list)}")

        if docs_list:
            df = convert_to_dataframe(docs_list)
            ids_to_drop = get_duplicated_docs_ids(df)

            if ids_to_drop:
                for idx, this_id in enumerate(tqdm.tqdm(ids_to_drop)):
                    try:
                        res = elastic_search.es_client.delete(
                            index=ES_INDEX,
                            id=this_id
                        )
                        logger.info(res)
                    except Exception as ex:
                        logger.error(f"Error deleting the doc: {traceback.format_exc()}")

        else:
            logger.info(f"NO THREADS RECEIVED FOR {dev_url}")

    logger.info("Process Complete.")
