import torch
import torch.nn as nn 
from torch.utils.data import Dataset , DataLoader , random_split
from torch.utils.tensorboard import SummaryWriter 
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from tqdm import tqdm
import warnings

from config import get_config , get_weights_file_path
from dataset import BilingualDataset , causal_mask
from model import Build_Transformer 

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]= "0"

def get_all_sentences(ds , lang) : 
    for item in ds : 
        yield item['translation'][lang]

def get_build_tokenizer(config,ds,lang) :
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) ## create relative path
    if not Path.exists(tokenizer_path) :     
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) ## if the tokenizer sees a word that it doesn't know it will replace it with 'UNK'
        tokenizer.pre_tokenizer = Whitespace() # Split by whitespace
        ## WordLevelTrainer -> it will split words using the white spaces 
        trainer = WordLevelTrainer(special_tokens=["[UNK]" , "[PAD]" , "[SOS]" , "[EOS]"] , min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds , lang ) , trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else : ## if the Tokenization is already done in a previous execution and the files exist 
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    raw_dataset = load_dataset('opus_books' , f'{config["lang_src"]}-{config["lang_tgt"]}' , split='train')
    
    ## Build the tokenizer 
    tokenizer_src = get_build_tokenizer(config , raw_dataset , config["lang_src"]) ## get the tokenized data for the source 
    tokenizer_tgt = get_build_tokenizer(config , raw_dataset , config["lang_tgt"]) ## get the tokenized data for the target 

    ## Splitting data : 90% training - 10% Validation
    train_ds_size = int(0.9*len(raw_dataset))
    val_ds_size   = int(len(raw_dataset) - train_ds_size)
    train_ds_raw , val_ds_raw = random_split(raw_dataset,[train_ds_size,val_ds_size])
    
    ## Getting the training data and validation data ready 
    ## train_ds and val_ds  have the format (encoder_input , decoder_input , encoder_mask , decoder_mask , target , src-text , tgt-text)
    train_ds = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])
    val_ds   = BilingualDataset(val_ds_raw  ,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])

    max_len_src =0
    max_len_tgt =0
    
    for item in raw_dataset :       
        src_ids = tokenizer_src.encode(item['translation'][config["lang_src"]]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config["lang_tgt"]]).ids
        
        max_len_src = max(max_len_src , len(src_ids))
        max_len_tgt = max(max_len_tgt , len(tgt_ids))
        
    print(f'Maximum length of source sentence : {max_len_src}')
    print(f'Maximum length of target sentence : {max_len_tgt}')
    
    train_dataloader = DataLoader(train_ds , batch_size=config['batch_size'], shuffle=True)
    val_dataloader   = DataLoader(val_ds , batch_size=1, shuffle=True)
    
    return train_dataloader , val_dataloader , tokenizer_src , tokenizer_tgt

def greedy_decode(model , source , source_mask , tokenizer_src , tokenizer_tgt , max_len ,device) : 
    sos_idx = tokenizer_src.token_to_id('[SOS]')
    eos_idx = tokenizer_src.token_to_id('[EOS]')
    
    ## Precompute the encoder output and reuse it for every token we get from the decoder 
    encoder_output = model.encode(source , source_mask) 
    
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True : 
        if decoder_input.size(1)==max_len : 
            break
        ## building the mask for the target 
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        ## output of the decoder 
        out = model.decode(encoder_output , source_mask , decoder_input , decoder_mask)
        ## get the next token 
        prob = model.project(out[:,-1])
        ## select the word with the max probability
        _,next_word = torch.max(prob,dim=-1)
        decoder_input = torch.cat([decoder_input , torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)] , dim=-1)            
        
        if next_word ==  eos_idx : 
            break
    return decoder_input.squeeze(0) ## squeeze to remove the batch dim 

        
def validation(model , val_ds, tokenizer_src , tokenizer_tgt , max_len , device , print_msg , global_step , writer , num_examples=2):
    model.eval()
    count = 0 
    src_txt   = []
    expected  = []
    predicted = []
    
    
    ## size of the control window
    console_width = 80 
    with torch.no_grad():
        for batch in val_ds : 
            count += 1 
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) ==1 ,  "Batch size must be 1 fr validation data "
            
            model_output = greedy_decode(model , encoder_input , encoder_mask , tokenizer_src , tokenizer_tgt , max_len , device)
            
            source_text = batch['src-text'][0]
            target_text = batch['tgt-text'][0]
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            
            
            src_txt.append(source_text)
            predicted.append(model_out_text)
            expected.append(target_text)
            
            
            ### Printing to the console
            print_msg('-'*console_width)
            print_msg(f'SOURCE : {source_text}') 
            print_msg(f'TARGET : {target_text}') 
            print_msg(f'PREDICTED: {model_out_text}')
            
            if count == num_examples : 
                break
                    
def get_model(config , vocab_src_len , vocab_tgt_len) : 
    model = Build_Transformer(vocab_src_len , vocab_tgt_len, config["seq_len"] , config["seq_len"],config['dim_model'])
    return model


def train_model(config):
    ## Defining the device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    
    Path(config["model_folder"]).mkdir(parents=True,exist_ok=True)
    
    train_dataloader , val_dataloader , tokenizer_src , tokenizer_tgt = get_dataset(config)
    
    model = get_model(config, tokenizer_src.get_vocab_size() , tokenizer_tgt.get_vocab_size()).to(device)
    ## Tensorboard (to visualize the loss )
    writer    = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters() , lr=config["lr"] , eps=1e-9)
    
    initial_epoch = 0
    global_step  = 0
    
    if config["preload"]:   
        model_filename = get_weights_file_path(config,config["preload"])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename) 
        initial_epoch = state["epoch"] +1 
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state["global_step"]
        
    loss_fn   = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]') , label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch,config['num_epochs']):
        batch_iterator = tqdm(train_dataloader,desc=f'Processing epoch {epoch:02d}')
        model.train()
        for batch in batch_iterator : 
            encoder_input = batch['encoder_input'].to(device) # (B,seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B,seq_len)
            encoder_mask  = batch['encoder_mask'].to(device)  # (B,1,1,seq_len)
            decoder_mask  = batch['decoder_mask'].to(device)  # (B,1,seq_len,seq_len)
            
            ## Running the tensors through transformers
            encoder_output = model.encode(encoder_input , encoder_mask  ) # (B,seq_len,dim_model)
            decoder_output = model.decode(encoder_output,encoder_mask,decoder_input, decoder_mask) # (B_seq_len , dim_model)
            proj_output    = model.project(decoder_output) # (B,seq_len,tgt_vocab_size)

             
            target = batch['target'].to(device) # (B,seq_len)
            # (B,seq_len,tgt_vocab_size) --> (B*seq_len,tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()) , target.view(-1))
            batch_iterator.set_postfix({f"loss":f"{loss.item():6.3f}"})
            
            ## Logging the loss
            writer.add_scalar('train loss' , loss.item() , global_step)
            writer.flush()
            
            loss.backward()
            ## update the weights 
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1
        validation(model , val_dataloader , tokenizer_src , tokenizer_tgt , config['seq_len'] , device , lambda msg:batch_iterator.write(msg) , global_step , writer )
        ## Save the model at each epoch 
        model_filename = get_weights_file_path(config,f'{epoch:02d}')
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'global_step':global_step
        },model_filename)
        
        
        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    print(config["lang_tgt"])
    train_model(config)  