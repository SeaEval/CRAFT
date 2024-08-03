
import fire
import logging

from datasets import load_from_disk
from transformers import AutoTokenizer


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

def main(
        region      : str = "",
        chunk_index : int = 0,
        model       : str = "meta-llama/Meta-Llama-3-70B-Instruct",
        chunk_size  : int = 1024,

        ):

    logger.info("")
    logger.info("Region: {}".format(region))
    logger.info("Model:  {}".format(model))
    logger.info("")

    # read the raw data
    data = load_from_disk('data/slimpajama/train')

    per_chunk_sample_size = 10000000
    data = data.select(range(chunk_index*per_chunk_sample_size, min(len(data),(chunk_index+1)*per_chunk_sample_size)))
    logger.info('This is from range {} to {}'.format(chunk_index*per_chunk_sample_size, min(len(data),(chunk_index+1)*per_chunk_sample_size)))

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    # read keywords
    with open("keywords/{}.txt".format(region), "r") as f:
        keywords = f.readlines()
    keywords = [item.lower().strip() for item in keywords]

    def contains_more_than_two_words(input_string, keywords):
        input_string = input_string.lower()  # Lowercase the input string
        count = 0
        for phrase in keywords:
            if phrase in input_string:
                count += 1
            if count > 2:
                return True
        return False


    def filter_keywords(examples):
        text            = examples['text'][0]
        tokenized_text  = tokenizer.encode(text, add_special_tokens=False)
        chunked_text_id = [tokenized_text[i:i+chunk_size] for i in range(0, len(tokenized_text), chunk_size)]
        all_text_chunk  = [tokenizer.decode(chunk) for chunk in chunked_text_id if len(chunk) > chunk_size//3]
        all_text_chunk  = [chunk for chunk in all_text_chunk if contains_more_than_two_words(chunk, keywords)]
        return {'text': all_text_chunk}

    # filter the data
    data = data.map(
        filter_keywords,
        batched           = True,
        batch_size        = 1,
        num_proc          = 64,
        writer_batch_size = 1000,
        remove_columns    = ['tokens', 'simhash'],
    )

    data.save_to_disk('data/{}/filtered_text/chunk_{}'.format(region, chunk_index))


if __name__ == "__main__":
    fire.Fire(main)