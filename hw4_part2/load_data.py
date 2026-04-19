import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

# Schema injection - provides table/column context to prevent hallucination
FLIGHT_SCHEMA = (
    "Tables: flight(flight_id, from_airport, to_airport, departure_time, arrival_time, "
    "airline_code, flight_days), "
    "airport_service(airport_code, city_code), "
    "city(city_code, city_name, state_code), "
    "airport(airport_code, airport_name, state_code), "
    "airline(airline_code, airline_name), "
    "fare(fare_id, from_airport, to_airport, fare_basis_code)"
)

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        self.split = split
        self.data_folder = data_folder
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.examples = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        with open(nl_path, 'r') as f:
            nl_queries = [line.strip() for line in f.readlines()]
        
        examples = []
        
        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            with open(sql_path, 'r') as f:
                sql_queries = [line.strip() for line in f.readlines()]
            
            for nl, sql in zip(nl_queries, sql_queries):
                nl_with_prefix = f"translate to SQL: {FLIGHT_SCHEMA} | query: {nl}"
                
                encoder_input = tokenizer(nl_with_prefix, return_tensors='pt', truncation=True, max_length=512)
                encoder_ids = encoder_input['input_ids'].squeeze(0)
                encoder_mask = encoder_input['attention_mask'].squeeze(0)
                
                decoder_input = tokenizer(sql, return_tensors='pt', truncation=True, max_length=1024)
                decoder_ids = decoder_input['input_ids'].squeeze(0)
                
                examples.append({
                    'encoder_ids': encoder_ids,
                    'encoder_mask': encoder_mask,
                    'decoder_ids': decoder_ids,
                })
        else:
            for nl in nl_queries:
                nl_with_prefix = f"translate to SQL: {FLIGHT_SCHEMA} | query: {nl}"
                
                encoder_input = tokenizer(nl_with_prefix, return_tensors='pt', truncation=True, max_length=512)
                encoder_ids = encoder_input['input_ids'].squeeze(0)
                encoder_mask = encoder_input['attention_mask'].squeeze(0)
                
                examples.append({
                    'encoder_ids': encoder_ids,
                    'encoder_mask': encoder_mask,
                })
        
        return examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [item['encoder_ids'] for item in batch]
    encoder_mask_list = [item['encoder_mask'] for item in batch]
    decoder_ids_list = [item['decoder_ids'] for item in batch]
    
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask_list, batch_first=True, padding_value=0)
    
    decoder_ids_padded = pad_sequence(decoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    
    # FIX: prepend BOS (PAD token = 0 = T5's decoder_start_token_id) so the model
    # learns to generate the FIRST SQL token from a clean start state.
    # Without this, decoder_inputs begins at tok1 (not PAD), so the first token
    # is never trained from the correct initial state.
    #
    # Correct layout:
    #   decoder_inputs:  [PAD, tok1, tok2, ..., tokN]
    #   decoder_targets: [tok1, tok2, ..., tokN, </s>]
    bos = torch.zeros(
        decoder_ids_padded.size(0), 1,
        dtype=decoder_ids_padded.dtype,
        device=decoder_ids_padded.device
    )
    decoder_with_bos = torch.cat([bos, decoder_ids_padded], dim=1)  # [PAD, tok1, ..., tokN, </s>]
    decoder_inputs = decoder_with_bos[:, :-1]   # [PAD, tok1, ..., tokN]
    decoder_targets = decoder_with_bos[:, 1:]   # [tok1, tok2, ..., </s>]
    
    # Initial decoder input for greedy/beam generation: just the BOS token
    initial_decoder_inputs = bos  # shape (B, 1)
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [item['encoder_ids'] for item in batch]
    encoder_mask_list = [item['encoder_mask'] for item in batch]
    
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask_list, batch_first=True, padding_value=0)
    
    batch_size = encoder_ids.size(0)
    initial_decoder_inputs = torch.full((batch_size, 1), PAD_IDX, dtype=torch.long)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    
    return train_x, train_y, dev_x, dev_y, test_x
