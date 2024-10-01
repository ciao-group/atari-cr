""" Borrowed from https://pytorch.org/tutorials/beginner/translation_transformer.html """

from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

from atari_cr.common.module_overrides import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK, PAD, BOS, EOS = 0, 1, 2, 3

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        """ 
        :param Tensor[B,N] tokens: Batch of indices
        """
        assert tokens.max() <= self.embedding.num_embeddings, f"Embedding index out of range: {tokens.max()} / {self.embedding.num_embeddings}"
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: Optional[int],
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 custom_src_embedder: Optional[nn.Module] = None):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = torch.nn.Sequential(
            nn.Linear(emb_size, tgt_vocab_size),
            nn.Softmax(dim=-1)
        )
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)
        
        # Modification to the module to allow custom src embedders
        self.custom_src_embedder = custom_src_embedder
        if custom_src_embedder:
            self.src_embedder = custom_src_embedder
        else:
            if src_vocab_size is None: 
                raise ValueError("Either src_vocab_size or custom_src_embedder need to be set")
            src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
            self.src_embedder = lambda src: self.positional_encoding(src_tok_emb(src))

    def forward(self,
                src: Tensor, # -> [B,S,W,H]
                tgt: Tensor, # -> [B,N]
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):

        src_emb = self.src_embedder(src) # -> [B,N_1,E]
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt)) # -> [B,N_2,E]
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
            src_padding_mask, tgt_padding_mask, memory_key_padding_mask) # -> [B,N_2,E]
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.src_embedder(src), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)
    
###
    
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    """ 
    :param Tensor[B,N] src_seq_len:
    :param Tensor[B,N] tgt_seq_len:
    """
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD)
    tgt_padding_mask = (tgt == PAD)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

###

def train_epoch(model: Seq2SeqTransformer, optimizer, train_loader, loss_fn):
    model.train()
    losses = 0

    for src, tgt in tqdm(train_loader):
        b = src.size(0)
        # src: Batch of 4-stacks of 84x84 greyscale images
        src = src.to(DEVICE) # -> [B,S,W,H]
        # tgt: Batch of tokens representing one gaze positions each; value between 0 and 7058
        tgt = tgt.to(DEVICE) # -> [B,N_2]

        # Apply the correct padding token
        tgt = torch.where(tgt == -83, PAD, tgt)

        # Cut off the last target token token
        tgt_input = tgt[:, :-1] # -> [B,N_2-1]

        # Create masks
        SRC_TOKEN_LENGTH = 17
        src_mask = torch.full([b,SRC_TOKEN_LENGTH], 100).to(DEVICE)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_mask, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        logits = logits.transpose(2,1) # -> [B,num_classes,N_2]

        # # Chose the highest probability as the class label; disabled because this must contain probabilites
        # logits = logits.argmax(dim=1) # -> [B,N_2-1]

        tgt_out = tgt[:, 1:] # -> [B,N_2-1]

        # # One hot encoding for tgt
        # tgt_out = F.one_hot(tgt_out.long(), num_classes=7058).transpose(2,1) # -> [B,num_classes,N_2-1]

        optimizer.zero_grad()
        loss = loss_fn(logits, tgt_out.long())
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_loader))

def evaluate(model, val_loader, loss_fn):
    model.eval()
    losses = 0

    for src, tgt in tqdm(val_loader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_loader))

###

def greedy_decode(model, src, src_mask, max_len):
    """ function to generate output sequence using greedy algorithm """
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(BOS).type(torch.long).to(DEVICE)
    for _ in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS:
            break
    return ys


# # actual function to translate input sentence into target language
# def translate(model: torch.nn.Module, src_sentence: str):
#     model.eval()
#     src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
#     num_tokens = src.shape[0]
#     src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
#     tgt_tokens = greedy_decode(
#         model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
#     return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")