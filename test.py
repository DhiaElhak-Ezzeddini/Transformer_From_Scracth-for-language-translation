from pathlib import Path
import torch 
import torch.nn as nn 
from config import get_config , get_weights_file_path 
from train_original import get_model , get_dataset , validation 


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

config = get_config()
print(config["lang_tgt"])
print(config["batch_size"])
train_dataloader , val_dataloader , tokenizer_src , tokenizer_tgt = get_dataset(config) ## Loading the data 
print(f"source tokenizer vocab size : {tokenizer_src.get_vocab_size()} ")
print(f"target tokenizer vocab size : {tokenizer_tgt.get_vocab_size()} ")
### Loading the model with Pretrained weights 
model = get_model(config, tokenizer_src.get_vocab_size() , tokenizer_tgt.get_vocab_size()).to(device)
model_filename =get_weights_file_path(config , f"19") ### f".."   number of the file tmodel... 
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])

### Running the Inference  
validation(model , val_dataloader , tokenizer_src , tokenizer_tgt , config['seq_len'], device  , lambda msg:print(msg) , 0 , None , num_examples=10)
