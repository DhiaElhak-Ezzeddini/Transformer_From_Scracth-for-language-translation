import torch
import torch.nn as nn 
from torch.utils.data import Dataset 
from typing import Any

class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len) ->None:
        super().__init__()
        self.ds = ds 
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang 
        self.tgt_lang = tgt_lang
        self.seq_len  = seq_len
        
        self.sos_token = torch.tensor([tokenizer_src.bos_token_id], dtype=torch.int64)  ## index of the SOS token in the source tokenized data 
        self.eos_token = torch.tensor([tokenizer_src.eos_token_id] , dtype=torch.int64) ## index of the EOS token in the source tokenized data 
        self.pad_token = tokenizer_src.pad_token_id                                     ## index of the PAD token in the source tokenized data 
        
    def __len__(self):
        return len(self.ds)
    def __getitem__(self,index: Any) -> Any:
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair['translation'][self.src_lang] ## raw source text // one sentence
        tgt_text = src_tgt_pair['translation'][self.tgt_lang] ## raw target data // one sentence

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids ## encoder input tokens (tokenization relative to the work done y the tokenizer implemented in the train.py file)
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids ## decoder input tokens 

        ## implementing the padding because the model works with a fixed sequence length 
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 
         
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0 : 
            raise ValueError("Sentence is too long")
        
        ## one sentence is sent as the input of the encoder 
        ## one sentence is sent as the input of the decoder
        ## one sentence is expected as the output of the decoder
        
         
        ## Adding SOS and EOS to the encoder input
        encoder_input = torch.cat( 
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token, 
                torch.full((enc_num_padding_tokens,), self.pad_token, dtype=torch.int64)
            ]
        )
        ## Adding sos to the decoder input 
        decoder_input = torch.cat( 
            [
                self.sos_token,
                torch.tensor(dec_input_tokens , dtype=torch.int64), 
                torch.full((dec_num_padding_tokens,), self.pad_token, dtype=torch.int64)
            ]
        )
        ## Adding eos to the target (the expected output of the decoder)
        target = torch.cat(
            [
                torch.tensor(dec_input_tokens , dtype=torch.int64) , 
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64 )
            ]
        )
        ### Confirming the size of the encoder decoder inputs and the target  
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert target.size(0) == self.seq_len
        
        return {
            "encoder_input" : encoder_input, # (seq_len)
            "decoder_input" : decoder_input, # (seq_len)
            "encoder_mask"  : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len) ==> Passing only the not padded tokens to the attention blocks
            "decoder_mask"  : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), 
            "target" : target , ## (seq_len)
            "src-text" : src_text,
            "tgt-text" : tgt_text
        }
## for the decoder we want each word to see only the words hat come before it 
## (key @ Query should be a lower triangular matrix) so we apply a mask (causal filter) to the decoder input 
        
def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size) , diagonal=1).type(torch.int) 
    return mask==0