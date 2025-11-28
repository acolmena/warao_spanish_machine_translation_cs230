from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import random
from tqdm.auto import tqdm, trange
import logging
import sys
import pandas as pd

"""
Different analyses on tokenizer to see how Warao text is being tokenized

Many of the analyses and functions are borrowed from David Dale
in his article: https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865

"""

random.seed(42)

def tok_analysis(model_name=None, dataset_path=None, warao_col=None, spanish_col=None):
    """
    Analyze tokenizer of a given pre-trained LM on Warao dataset.
    We will use only our Warao training dataset
    
    Args:
        model_name (str): The name of the pre-trained,

    """
    # set up logger
    logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load dataset
    data_files = {'train': dataset_path}
    dataset = load_dataset('csv', data_files=data_files)['train']
    logger.info(f"\nLoaded dataset from {dataset_path}")
    print(f"\nDataset preview: {dataset[:2]}")

    # Tokenization analysis
    token_lengths = []
    all_tokens = []
    for example in dataset:
        sentence = example[warao_col]
        tokens = tokenizer.tokenize(sentence)
        all_tokens += tokens
        token_lengths.append(len(tokens))

    # breakpoint()
    all_uniq_tokens = list(set(all_tokens))
    print(f"\nTotal unique tokens in Warao dataset: {len(all_uniq_tokens)}")
    print(f"\nPreview of unique tokens: {all_uniq_tokens[:5]}")
    series_tokens = pd.Series(all_tokens)
    series_tok_lens = series_tokens.apply(len)
    print(series_tokens.describe())
    print(series_tokens.value_counts())
    print(series_tok_lens.describe())

    
    # determine how many unknown tokens we're getting 
    texts_with_unk = [
        text for text in tqdm(dataset[warao_col], desc="Checking for unknown tokens") 
        if tokenizer.unk_token_id in tokenizer(text).input_ids
    ]

    num_warao_unk_toks = len(texts_with_unk)
    logger.info(f"\n{"Number of Warao sentences with unknown tokens: {num_warao_unk_toks}" if num_warao_unk_toks > 0 else "No unknown tokens generated for Warao sentences"}")


    # avg_token_length = sum(token_lengths) / len(token_lengths)
    # print(f"Average token length for {model_name} on {dataset_path}: {avg_token_length}")

    return 

if __name__ == "__main__":
    model_name = "facebook/m2m100_418M"
    # read in from Google Sheets 
    dataset_path = "./input/parallel_train.csv"
    warao_col = 'warao_sentence'
    spanish_col = 'spanish_sentence'

    tok_analysis(model_name, dataset_path, warao_col, spanish_col)