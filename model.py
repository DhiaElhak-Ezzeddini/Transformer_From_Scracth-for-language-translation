#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@@@ Building the core of the model @@@#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
import torch 
import torch.nn as nn
import math 

## Starting by the @@@@ input embedding @@@@ 
class InputEmbedding(nn.Module):
    def __init__(self,dim_model:int , vocab_size:int): 
        super().__init__()
        self.dim_model = dim_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,dim_model)  
    
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.dim_model)

## Positional Encoding ==> vectors of size dim_model , computed only once 
## and reused for every sentence during training and inference 

class PositionalEncoding(nn.Module):
    def __init__(self , dim_model:int , seq_len:int , dropout:float ) -> None : ## seq_len : maximum length of a sentence 
        super().__init__()
        self.dim_model = dim_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        ## matrix of shape (seq_len,dim_model) 
        pos_enc = torch.zeros(seq_len,dim_model)
        
        ## vector of shape (seq_len,1)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        
        ## denominator of the formula 
        div_term = torch.exp(torch.arange(0,dim_model,2).float() * (-math.log(10000.0) / dim_model)) 
        ## sine to even positions 
        pos_enc[:,0::2] = torch.sin(position*div_term)
        ## cosine to odd position
        pos_enc[:,1::2] = torch.cos(position * div_term)
        
        ## shape (1,seq_len,dim_model) // because we have a batch of sentences 
        pos_enc = pos_enc.unsqueeze(0) 

        self.register_buffer('pos_enc' , pos_enc) ## to be saved (not like a parameter of the model) ==> the tensor will be saved in the buffer 
        
    def forward(self,x) : 
        x = x + (self.pos_enc[:, :x.shape[1] ,:]).requires_grad_(False) ## Not to be learned // x : embedded input
        return self.dropout(x)

## Normalization Layer ( add&norm )

class LayerNormalization(nn.Module) : 
    def __init__(self,eps:float=10**-6) -> None : 
        super().__init__()
        self.eps = eps ## to avoid dividing by zero
        self.alpha = nn.Parameter(torch.ones(1)) ## Multiplied ==> to be learned 
        self.bias  = nn.Parameter(torch.ones(1)) ## Added ==> to be learned  
        
    def forward(self,x):
        mean = x.mean(dim=-1 , keepdim=True)
        std = x.std(dim=-1 ,keepdim=True) 
        ## formula 
        return self.alpha * (x - mean) / (std + self.eps) + self.bias 
 
## Feed Forward layer 
class FeedForwardLayer(nn.Module) : ### ==> explore more features in a higher dimension space 
    def __init__(self, dim_ff:int , dim_model:int , dropout:float) : 
        super().__init__()
        
        self.linear_1 = nn.Linear(dim_model,dim_ff)  
        self.dropout  = nn.Dropout(dropout)         
        self.linear_2 = nn.Linear(dim_ff,dim_model) 
        
    def forward(self,x) :
        # (batch , seq_len , d_model) --> (batch , seq_len , dff) --> (batch , seq_len , d_model)
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x 
    
######################################
##### Multi-Head Attention Block #####
######################################

class MultiHeadAttention(nn.Module) : 
    def __init__(self,dim_model:int , h:int , dropout:float) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.h = h
        assert dim_model % h == 0 , "dim_model is not divisible by h (number of heads) !!" 
        self.d_k = dim_model // h
        self.w_q = nn.Linear(dim_model,dim_model)   ## Matrix W for the Queries 
        self.w_k = nn.Linear(dim_model,dim_model)   ## Matrix W for the Keys 
        self.w_v = nn.Linear(dim_model,dim_model)   ## Matrix W for the Values
        self.w_o = nn.Linear(dim_model,dim_model)   ## Matrix W for output projection layer 
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod ## we can call it without having an instance of the MultiHeadAttention class
    def attention(query , key , value, mask,dropout:nn.Dropout) : 
        d_k = query.shape[-1] ## last dim of the query (the model dimension of the specific head)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k) ## matrix multiplication ===> scaled dot product attention 
        if mask is not None : 
            attention_scores.masked_fill_(mask==0,-1e9) ## softmax(-1e9) ~ 0.0 
            
        attention_scores = attention_scores.softmax(dim=-1) ## (batch , h , seq_len , seq_len ) --> (batch , h , seq_len , ses_len) 
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return (attention_scores @ value) , attention_scores ## (attention_scores @ value) : (batch , h , seq_len , seq_len ) --> (batch , h , seq_len , d_k) 

    def forward(self,q,k,v,mask) : # if wa want some words to do not interact with other words we mask them 
        query = self.w_q(q) # (batch , seq_len , d_model) --> (batch , seq_len , d_model)
        key   = self.w_k(k) # (batch , seq_len , d_model) --> (batch , seq_len , d_model)
        value = self.w_v(v) # (batch , seq_len , d_model) --> (batch , seq_len , d_model)
        
        ## (batch , seq_len , dim_model) --> (batch ,seq_len , h , d_k ) --> (batch , h , seq_len , d_k ) 
        ## each head should see all the sentence but with a smaller part of the embedding of the words ==> we split only the embedding dimension 
        ## than we transpose h and seq_len 
        query = query.view(query.shape[0] , query.shape[1] , self.h , self.d_k).transpose(1,2) # (batch , h , seq_len , d_k ) 
        key   = key.view(key.shape[0] , key.shape[1] , self.h , self.d_k).transpose(1,2)       # (batch , h , seq_len , d_k ) 
        value = value.view(value.shape[0] , value.shape[1] , self.h , self.d_k).transpose(1,2) # (batch , h , seq_len , d_k ) 
        
        
        x , self.attention_scores = MultiHeadAttention.attention(query,key,value,mask,self.dropout)
        ## (batch , h , seq_len , d_k ) --> (Batch , seq_len , h , d_k) --> ## (batch , seq_len , dim_model= h * d_k)
        x = x.transpose(1,2).contiguous().view(x.shape[0] , -1 , self.h*self.d_k)
        
        ## (batch , seq_len , d_model) --> (batch , seq_len , d_model)
        return self.w_o(x)

        
class ResidualConnectionLayer(nn.Module) : 
    def __init__(self,dropout:float) -> None:
        super().__init__() 
        self.dropout = nn.Dropout(dropout)
        self.norm    = LayerNormalization()
        
    def forward(self,x , sublayer) : 
        return x + self.dropout(sublayer(self.norm(x)))
    
    
######################################
##### Building the Encoder Block #####
######################################

class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttention,feed_forward_layer:FeedForwardLayer , dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_layer = feed_forward_layer
        self.residual_connections = nn.ModuleList([ResidualConnectionLayer(dropout) for _ in range(2)]) 

    def forward(self,x,src_mask):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x,src_mask)) ## the input itself plays the role of Q , K and V 
        x = self.residual_connections[1](x,self.feed_forward_layer) 
        return x 
    
class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self,x,mask):
        for layer in self.layers :
            x = layer(x,mask)
        return self.norm(x)
    
 ####################################
 ##### Building the decoder #########
 ####################################   
    
class DecoderBlock(nn.Module) : 
    def __init__(self,self_attention_block: MultiHeadAttention , cross_attention_block : MultiHeadAttention , feed_forward_block : FeedForwardLayer , dropout:float) -> None:
         super().__init__()
         self.self_attention_block  = self_attention_block
         self.cross_attention_block = cross_attention_block
         self.feed_forward_block = feed_forward_block
         self.residual_connections = nn.ModuleList([ResidualConnectionLayer(dropout) for _ in range(3)])

    def forward(self,x,encoder_output , src_mask , tgt_mask): ## tgt_mask : the mask applied to the decoder // src_mask : the mask applied to the encoder
        x = self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,tgt_mask)) ## the self-attention block of the decoder 
        x = self.residual_connections[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output , src_mask)) ## the cross attention block of the decoder
        x = self.residual_connections[2](x,self.feed_forward_block)
        
        return x 

## decoder is N times the DecoderBlock , one after another  
class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self , x , encoder_output , src_mask , tgt_mask ):
        for layer in self.layers :
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self,dim_model,vocab_size)->None:
        super().__init__()
        self.proj = nn.Linear(dim_model,vocab_size)
    def forward(self,x):
        ## (Batch, seq_len , dim_model) => (Batch , seq_len , vocab_size)
        x = self.proj(x)
        x = torch.log_softmax(x , dim=-1) ## probability across all the vocabulary size (across all the tokens in the vocabulary to predict the next token)   
        return x 
    
     
 #############################################
 #############################################
 ####### Building the Transformer Block ######
 ############################################# 
 #############################################

class Transformer(nn.Module):
    def __init__(self , encoder:Encoder , decoder:Decoder , src_embed:InputEmbedding , tgt_embed:InputEmbedding , src_pos:PositionalEncoding , tgt_pos:PositionalEncoding , proj_layer:ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer
        
    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)

    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

    def project(self,x) : 
        return self.proj_layer(x)
    
    
def Build_Transformer(src_vocab_size:int , tgt_vocab_size:int , src_seq_len:int , tgt_seq_len:int , dim_model:int=512 ,N:int=6, h:int=8,dropout:float=0.1,d_ff:int=2048) -> Transformer :
    
     ## creating the embedding layers 
    src_embed = InputEmbedding(dim_model , src_vocab_size)
    tgt_embed = InputEmbedding(dim_model , tgt_vocab_size)

     ## Creating the positional encoder
    src_pos = PositionalEncoding(dim_model , src_seq_len , dropout)
    tgt_pos = PositionalEncoding(dim_model , tgt_seq_len , dropout)

     ## Creating the encoder blocks
    encoder_blocks = []
    for _ in range(N) : 
        encoder_self_attention_block = MultiHeadAttention(dim_model,h,dropout)
        feed_forward_block = FeedForwardLayer(d_ff,dim_model,dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block , feed_forward_block , dropout)
        encoder_blocks.append(encoder_block)
    ## Creating the decoder blocks
    decoder_blocks = []
    for _ in range(N) : 
        decoder_self_attention_block  = MultiHeadAttention(dim_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttention(dim_model,h,dropout)
        feed_forward_block = FeedForwardLayer(d_ff,dim_model,dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block , feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)
         
    ## Creating the decoder and the encoder 
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Creating the projection Layer
    proj_layer = ProjectionLayer(dim_model , tgt_vocab_size)
    
    # Creating the transformer: 
    transformer = Transformer(encoder , decoder , src_embed , tgt_embed, src_pos, tgt_pos, proj_layer)
    
    ## Initializing the parameters so they don't start with random variables
    for p in transformer.parameters() :
        if p.dim() > 1 : 
            nn.init.xavier_uniform_(p)
            
    return transformer