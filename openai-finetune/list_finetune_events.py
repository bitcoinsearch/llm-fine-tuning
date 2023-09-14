import openai
from pprint import pprint
import os
from dotenv import load_dotenv
import warnings
from loguru import logger

warnings.filterwarnings("ignore")
load_dotenv()

OPENAI_ORG_KEY = os.getenv("OPENAI_ORG_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.organization = OPENAI_ORG_KEY
openai.api_key = OPENAI_API_KEY

# Enter fine-tuning job id (i.e., "ftjob-8fVPvtgokefrgSFRdvyGIAzc")
FINETUNING_JOB_ID = "ftjob-8fVPvtgokefrgSFRdvyGIAzc"


if __name__ == "__main__":

    try:
        # retrieve fine-tune status
        retrieved_ft = openai.FineTuningJob.retrieve(FINETUNING_JOB_ID)

        if retrieved_ft["status"] == "succeeded":
            logger.info(retrieved_ft)
            logger.success("Fine-Tuning Successful!")
            logger.success(f"Fine-tuned model name: {retrieved_ft['fine_tuned_model']}")

        else:
            # logger.info(retrieved_ft)
            pprint(openai.FineTuningJob.list_events(FINETUNING_JOB_ID))
            logger.info(f"Fine-Tuning Status: {retrieved_ft['status']}")

    except Exception as ex:
        logger.error(ex)

    # # to delete uploaded file
    # file_name = "file-U0gadd9sL2JAShVh8oEaKZXJ"
    # pprint(openai.File.delete(file_name))
    # pprint(openai.File.list())

    # # to cancel finetune job
    # cancel_ft = openai.FineTuningJob.cancel(FINETUNING_JOB_ID)
    # logger.info(cancel_ft)
