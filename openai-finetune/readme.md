## OpenAI Fine-tuning

This directory provides scripts and guides to finetune the OpenAI pre-trained models for your specific needs.

### Usage:
1. `finetune.py`: This file consists of all the steps required for fine-tuning the pre-trained model by following the below steps:  
   * **Data Collection:** Fetch the data from elasticsearch index
   * **Data Preparation:** Prepare the custom training data in the required format `jsonlines`
   * **Validate Data:** Check whether the prepared data files is correctly formatted. (using `jsonl_data_stats.py`)
   * **Provide Estimates:** Provide cost estimates for fine-tuning based on the dataset. (using `jsonl_data_stats.py`)
   * **Upload the File:** Upload the file for fine-tuning using OpenAI API
   * **Create fine-tuning job:** Begin the fine-tuning job based on provided fine-tuning model.
   * **Retrieve fine-tune status:** Retrieve fine-tune status once after 60s, so you can get `finetune_id` which you further use to get fine-tuning status later on.

2. `list_finetune_events.py`: Configure fine-tuning job id in this file, and you will be able to retrieve the fine-tuning status and events of an ongoing fine-tuning.

### References:
   * API: https://platform.openai.com/docs/api-reference/fine-tuning
   * Documentation: https://platform.openai.com/docs/guides/fine-tuning