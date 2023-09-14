import json
import tiktoken
import numpy as np
from collections import defaultdict
from loguru import logger


def check_jsonl_data_format(file_path):
    # Load the data
    with open(file_path) as f:
        dataset = [json.loads(line) for line in f]

    # Initial dataset stats
    logger.info(f"Num examples: {len(dataset)}")

    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            if not content or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        logger.warning("Found errors:")
        for k, v in format_errors.items():
            logger.warning(f"{k}: {v}")
            return False
    else:
        logger.success("No errors found")
        return True


def get_json_cost(file_path, n_epochs=3):
    num_tokens = 0

    with open(file_path) as f:
        dataset = [json.loads(line) for line in f]

    encoding = tiktoken.get_encoding("cl100k_base")

    for d in dataset:
        num_tokens += len(encoding.encode(str(d)))

    print(num_tokens)

    # rough estimate # base cost per 1k tokens * number of tokens in the input file * number of epochs trained
    print(0.0080 * num_tokens * n_epochs)


def get_jsonl_data_stats(file_path, n_epochs):
    # Load the data
    with open(file_path) as f:
        dataset = [json.loads(line) for line in f]

    # Beyond the structure of the message, we also need to ensure that the length does not exceed the 4096 token limit.

    # Token counting functions
    encoding = tiktoken.get_encoding("cl100k_base")

    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens

    def print_distribution(values, name):
        logger.info(
            f"Distribution of {name}: \t1). min/max: {min(values)}, {max(values)} \t2). mean/median: {np.mean(values)}, {np.median(values)} \t3). p5/p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

    # Last, we can look at the results of the different formatting operations before proceeding with creating a fine-tuning job:

    # Warnings and tokens counts
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]

        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1

        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    logger.info(f"Num examples missing system message: {n_missing_system}")
    logger.info(f"Num examples missing user message: {n_missing_user}")

    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")

    n_too_long = sum(l > 4096 for l in convo_lens)
    if n_too_long == 0:
        logger.success("No examples are over the 4096 token limit.")
    else:
        logger.warning(
            f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning")

    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = 4096

    # MIN_TARGET_EXAMPLES = 100
    # MAX_TARGET_EXAMPLES = 25000
    # TARGET_EPOCHS = 3
    # MIN_EPOCHS = 1
    # MAX_EPOCHS = max_epochs

    # n_epochs = TARGET_EPOCHS
    # n_train_examples = len(dataset)
    # if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    #     n_epochs = min(MAX_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    # elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    #     n_epochs = max(MIN_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    logger.info(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
    logger.info(f"You'll train for {n_epochs} epochs on this dataset")
    logger.info(f"You'll be charged for ~{n_billing_tokens_in_dataset * n_epochs} tokens")

    # estimated_cost = base cost per 1k tokens * number of tokens in the input file * number of epochs trained
    estimated_cost = 0.0080 * n_billing_tokens_in_dataset * n_epochs
    logger.success(
        f"Estimated cost for this dataset is: $ {estimated_cost}, (GPT-3.5 Turbo at training cost of $0.0080/1K tokens)")


if __name__ == "__main__":
    check_jsonl_data_format(file_path="../data.jsonl")
    get_jsonl_data_stats(file_path="../data.jsonl", n_epochs=3)
    # get_json_cost(file_path="../data.jsonl")
