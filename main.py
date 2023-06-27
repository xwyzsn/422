import torch
from torch import nn
from torch import einsum
from einops import rearrange, repeat
import torch.nn.functional as F
from torch.optim import SGD, Adam
import math
from torch.optim.optimizer import Optimizer
from data import tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np
from data import CustomDataset
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class LinearAttention(nn.Module):
    def __init__(self, dim: int, embed_size: int, head: int):
        super().__init__()
        self.embed_size = embed_size
        self.head = head
        self.head_dim = embed_size // head
        assert self.head_dim * head == embed_size, "emvbed size can not divide by head"
        self.scale = embed_size ** -0.5
        self.to_q = nn.Linear(dim, embed_size)
        self.to_k = nn.Linear(dim, embed_size)
        self.to_v = nn.Linear(dim, embed_size)
        self.to_out = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v, mask=None):
        q = self.to_q(q)  # q: batch_size,seq_len,embed_size
        k = self.to_k(k)  # k: batch_size,seq_len,embed_size
        v = self.to_v(v)  # v: batch_size,seq_len,embed_size
        # q k 
        q, k, v = map(lambda x: rearrange(x, 'b s (h e) -> b h s e', h=self.head, e=self.head_dim), [q, k, v])
        energy = einsum('b h s e, b h k e->b h s k', q, k)
        energy = self.scale * energy  # batch,head,seq_len,key_len
        if mask is not None:
            tri_mask, padding_mask = mask
            padding_mask = ~padding_mask
            padding_mask = repeat(padding_mask, 'b s -> b h s k', h=self.head, k=energy.size(2))  # batch,head,seq_len
            padding_mask = rearrange(padding_mask, 'b h s k -> b h k s')
            energy.masked_fill_(padding_mask.bool(), torch.finfo(q.dtype).min)  # b h seq_len seq_len
            if tri_mask is not None:
                tri_mask = ~tri_mask.bool()
                energy.masked_fill_(tri_mask.bool(), torch.finfo(q.dtype).min)
        energy = energy.softmax(dim=-1)
        attn = einsum('b h s k,b h k e->b h s e', energy, v)
        out = rearrange(attn, 'b h s e -> b s (h e)')
        out = self.to_out(out)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class LayerNorm(nn.Module):

    def __init__(self, embed_size, drop_out=0.1):
        super().__init__()
        self.drop_out = 0.1
        self.ln = nn.LayerNorm(embed_size)
        self.drop = nn.Dropout(drop_out)

    def forward(self, x):
        return self.drop(self.ln(x))


class FeedForward(nn.Module):
    def __init__(self, dim_in, drop_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_in * 4)
        self.fc2 = nn.Linear(dim_in * 4, dim_in)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        return self.dropout(self.fc2(self.relu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, dim_in: int, embed_dim: int, head: int, drop_rate=0.1):
        super().__init__()

        self.attn = LinearAttention(dim_in, embed_dim, head)
        self.laynorm1 = LayerNorm(embed_dim)
        self.laynorm2 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, drop_rate)

    def forward(self, x, mask=None):
        attn = self.attn(x, x, x, mask) + x
        attn = self.laynorm1(attn)
        out = self.ff(attn) + attn
        out = self.laynorm2(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: batch_size,seq_len,embed_size
        batch_size, seq_len, embed_size = x.size(0), x.size(1), x.size(2)
        device = x.device
        positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        positions = repeat(positions, 's d->b s d', b=batch_size)
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * (-math.log(10000.0) / self.dim))
        PE = torch.zeros((batch_size, seq_len, embed_size))
        PE[:, :, 0::2] = torch.sin(positions * div_term)  # batch,seq_len/2,1, 1 time
        PE[:, :, 1::2] = torch.cos(positions * div_term)
        PE = PE.to(device=device)
        return PE


class Encoder(nn.Module):

    def __init__(self, dim_in, embed_size, head, drop_rate, num, vob_size):
        super().__init__()
        self.net = nn.ModuleList(
            [TransformerBlock(dim_in=dim_in, embed_dim=embed_size, head=head, drop_rate=drop_rate) for _ in range(num)]
        )
        self.PE = PositionalEncoding(dim=embed_size)
        self.embed_size = embed_size
        self.embed = nn.Embedding(num_embeddings=vob_size, embedding_dim=embed_size, padding_idx=0)

    def forward(self, x, mask=None):
        input_embed = self.embed(x) * (self.embed_size ** (-0.5))
        x = self.PE(input_embed) + input_embed
        for idx in range(len(self.net)):
            x = self.net[idx](x, mask)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, dim_in, embed_size, head, enc_out_dim, drop_rate):
        super().__init__()
        self.attn1 = LinearAttention(dim=dim_in, embed_size=embed_size, head=head)
        self.attn2 = LinearAttention(dim=enc_out_dim, embed_size=embed_size, head=head)
        self.laynorm1 = LayerNorm(embed_size)
        self.laynorm2 = LayerNorm(embed_size)
        self.ff = FeedForward(dim_in=embed_size, drop_rate=drop_rate)

    def generate_mask(self, seq_len):
        return torch.tril(torch.ones(size=(seq_len, seq_len)), diagonal=0)

    def forward(self, x, enc_out, mask=None):
        batch_size, seq_len, embed_size = x.size(0), x.size(1), x.size(2)
        device = x.device
        tri_mask = self.generate_mask(seq_len=seq_len).to(device=device)  # seq_len,seq_len
        # mask_1 = tri_mask + mask
        attn = self.laynorm1(self.attn1(x, x, x, (tri_mask, mask)) + x)
        attn = self.laynorm2(self.attn2(attn, enc_out, enc_out, (None, mask)) + attn)
        out = self.ff(attn) + attn
        return out


class Decoder(nn.Module):
    def __init__(self, dim_in, embed_size, head, vob_size, drop_rate: float = 0.1, num: int = 3) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.net = nn.ModuleList(
            [DecoderBlock(dim_in=dim_in, embed_size=embed_size, head=head, enc_out_dim=embed_size, drop_rate=drop_rate)
             for _ in range(num)]
        )
        self.PE = PositionalEncoding(dim=embed_size)
        self.embed = nn.Embedding(num_embeddings=vob_size, embedding_dim=embed_size, padding_idx=0)

    def forward(self, output_ids, enc_out, mask=None):
        output_embed = self.embed(output_ids) * (self.embed_size ** (-0.5))
        x = self.PE(output_embed) + output_embed
        for idx in range(len(self.net)):
            x = self.net[idx](x, enc_out, mask)
        return x


class Transformer(nn.Module):

    def __init__(self,
                 dim_in: int,
                 vocab_size: int,
                 head: int,
                 num: int,
                 embed_size: int,
                 drop_out: int = 0.1,
                 device: str = 'cuda'):
        super().__init__()
        self.fc = nn.Linear(embed_size, vocab_size)
        self.encoder = Encoder(dim_in, embed_size, head, drop_rate=0.1, num=num, vob_size=vocab_size)
        self.decoder = Decoder(dim_in=dim_in, embed_size=embed_size, head=head, drop_rate=drop_out, vob_size=vocab_size,
                               num=num)
        self.PE = PositionalEncoding(dim=embed_size)
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
        self.device = device

    def make_input_mask(self, input):
        # return torch.stack([torch.tensor(item.get('attention_mask')) for item in input]).to(self.device)
        return input.get('attention_mask').to(self.device)

    def make_output_mask(self, out_seq):
        # return torch.stack([torch.tensor(item.get('attention_mask')) for item in out_seq]).to(self.device)
        return out_seq.get('attention_mask').to(self.device)

    def get_inputOroutput_ids(self, x):
        if isinstance(x, list):
            return torch.stack([torch.tensor(item.get('ids')) for item in x]).to(device=self.device)
        else:
            return torch.tensor(x.get('ids')).to(device=self.device)

    def shift_right(self, out_seq: dict):
        ids = out_seq['ids']  # batch_size,seq_len
        ids[:, -1] = tokenizer.token_to_id('<BOS>')
        attention_mask = out_seq['attention_mask']
        attention_mask[:, -1] = 1
        ids = torch.roll(ids, 1, -1)
        attention_mask = torch.roll(attention_mask, 1, -1)
        return {'ids': ids, 'attention_mask': attention_mask}

    def forward(self, input_seq, out_seq):
        input_ids = self.get_inputOroutput_ids(input_seq)
        out_seq = self.shift_right(out_seq)
        output_ids = self.get_inputOroutput_ids(out_seq)
        input_mask = self.make_input_mask(input_seq).bool()
        enc_out = self.encoder(input_ids, (None, input_mask))
        masks = self.make_output_mask(out_seq).bool()
        x = self.decoder(output_ids, enc_out, masks)
        out = self.fc(x)
        out = F.softmax(out, dim=-1)
        return out


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask=None, ignore_idx=0):  # batch,seq_len,pro
        input = input.view(-1, input.size(-1))
        target = target.view(-1)

        # if mask is not None:
        #     mask = (mask.view(-1).bool())
        #     input = input[mask]
        #     target = target[mask]

        loss = F.cross_entropy(input=input, target=target, ignore_index=ignore_idx, label_smoothing=0.1)

        return loss


class Model(nn.Module):
    def __init__(self, device, vocab_size, embed_size):
        super().__init__()
        self.transformer = nn.Transformer(device=device, batch_first=True)
        self.fc = nn.Linear(embed_size, vocab_size).to(device)
        self.PE = PositionalEncoding(dim=embed_size).to(device)
        self.device = device
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0).to(device)

    def shift_right(self, ids, mask):
        # ids:batch,seq_len
        # mask : batch ,sen_len
        start_token = tokenizer.token_to_id('<BOS>')
        ids[:, -1] = start_token
        mask[:, -1] = 1
        ids = torch.roll(ids, 1, -1)
        mask = torch.roll(mask, 1, -1)
        return ids, mask

    def forward(self, src, tar):
        input_ids = src.get('ids').to(self.device)  # batch,seq_len,embed_size
        mask = src.get('attention_mask').to(self.device)
        out_ids = tar.get('ids').clone().to(self.device)
        out_mask = tar.get('attention_mask').clone().to(self.device)
        out_ids, out_mask = self.shift_right(out_ids, out_mask)
        src_embed = self.embed(input_ids)
        src_embed = src_embed + self.PE(src_embed)
        tar_embed = self.embed(out_ids)
        tar_embed = tar_embed + self.PE(tar_embed)
        trg_mask = self.transformer.generate_square_subsequent_mask(input_ids.size(1)).to(device=self.device)
        out = self.transformer(src_embed, tar_embed, src_key_padding_mask=~mask.bool(),
                               tgt_key_padding_mask=~out_mask.bool(), tgt_mask=trg_mask)
        out = F.softmax(self.fc(out), dim=-1)
        return out, out_ids, out_mask


class Trainer:

    def __init__(self, dim, vocab_size, head, num, embed_size, dataset: Dataset
                 , batch_size, config: dict):
        # self.transformer = Transformer(dim_in=dim,vocab_size=vocab_size,head=head,num=num,embed_size=embed_size).to(device=config['device'])
        self.CE = CrossEntropyLoss()
        self.model = Model(embed_size=embed_size, vocab_size=vocab_size, device=config['device'])
        self.opt = Adam(self.model.parameters(), betas=(0.9, 0.98),
                        eps=1.0e-9, lr=5e-4)  # SGD(self.transformer.parameters(),lr=1.0)
        self.dataset = dataset
        self.device = config['device']
        train_size = int(0.8 * len(dataset))
        # trainset,testset = random_split(self.dataset,[train_size,len(self.dataset)-train_size])
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.config = config
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.opt,factor=0.1,patience=5,verbose=True
        )

    def train_loop(self):
        for epoch in range(self.config['epoch']):
            losses = []
            for batch_idx, (src, tar) in enumerate(self.dataloader):
                out_ids, out_mask = tar.get('ids').to(self.device), tar.get('attention_mask').to(self.device)
                out, _, _ = self.model(src, tar)
                loss = self.compute_loss(out, target=out_ids, mask=out_mask)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                losses.append(loss.item())
            # self.tranlate_sentence()
            self.lr_scheduler.step(np.mean(losses))
            print(f"{epoch=}  avg loss {np.mean(losses)}")


    def compute_loss(self, input, target, mask=None):
        loss = self.CE(input, target, mask)
        return loss

    def train(self):
        for epoch in range(self.config['epoch']):
            losses = []
            for batch_idx, (src, tar) in enumerate(self.dataloader):
                input_ids = tar.get('ids').to(self.config['device'])
                mask = src.get('attention_mask').to(self.config['device'])
                out = self.transformer(src, src)

                loss = self.compute_loss(out, target=input_ids, mask=mask)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                # self.lr_scheduler.step()
                losses.append(loss.item())
            print(f"{epoch=}  avg loss {np.mean(losses)}")
            self.tranlate_sentence()

    @torch.no_grad()
    def translate(self, src, tar, max_length=10):
        device = 'cpu'
        input_ids = src.get('ids').to(device)
        input_mask = src.get('attention_mask').to(device)
        self.model = self.model.to(device)
        input_embed = self.model.embed(input_ids)
        input_embed = self.model.PE(input_embed) + input_embed
        enc_out = self.model.transformer.encoder(input_embed, src_key_padding_mask=~input_mask.bool())
        for i in range(1, max_length):
            out_ids = tar.get('ids').to(device)
            out_mask = tar.get('attention_mask').to(device)
            out_embed = self.model.embed(out_ids)
            out_embed = self.model.PE(out_embed) + out_embed
            x = self.model.transformer.decoder(out_embed, enc_out, tgt_key_padding_mask=~out_mask.bool(),
                                               memory_key_padding_mask=~input_mask.bool())
            out = self.model.fc(x)
            max_ids = torch.argmax(out, dim=-1)
            tar.get('ids')[0, i] = max_ids[0, i]
            tar.get('attention_mask')[0, i] = 1
        return tar

    @torch.no_grad()
    def inference(self, src, tar, max_length=10):
        input_ids = self.transformer.get_inputOroutput_ids(src)
        input_mask = self.transformer.make_input_mask(src).bool()
        enc_out = self.transformer.encoder(input_ids, (None, input_mask))

        for i in range(1, max_length):
            # output_embed = self.transformer.embed(self.transformer.get_inputOroutput_ids(tar))
            # output_embed = self.transformer.PE(output_embed) + output_embed
            output_ids = self.transformer.get_inputOroutput_ids(tar)
            masks = self.transformer.make_output_mask(tar).bool()
            x = self.transformer.decoder(output_ids, enc_out, masks)
            out = self.transformer.fc(x)
            out = F.softmax(out, dim=-1)
            max_ids = torch.argmax(out, dim=-1)
            tar.get('ids')[0, i] = max_ids[0, i]
            tar.get('attention_mask')[0, i] = 1
        return tar

    def tranlate_sentence(self):
        encoding = tokenizer.encode("hello,what's your name")
        src = {'attention_mask': torch.tensor(encoding.attention_mask).unsqueeze_(0),
               'ids': torch.tensor(encoding.ids).unsqueeze_(0)}
        tar_enc = tokenizer.encode("<BOS>")
        ids, atttention_mask = tar_enc.ids, tar_enc.attention_mask
        ids[1], atttention_mask[1] = 0, 0
        tar = {'attention_mask': torch.tensor(atttention_mask).unsqueeze_(0), 'ids': torch.tensor(ids).unsqueeze_(0)}
        # tar_out = self.inference(src=src,tar=tar)
        tar_out = self.translate(src=src, tar=tar)
        tar_out = tar_out.get('ids').squeeze_(0)
        o = tokenizer.decode(tar_out.tolist())
        print(o)
        return o


if __name__ == '__main__':
    # x1 = torch.randint(size=(16,128),low=0,high=10,device='cuda')
    # x2 = torch.randint(size=(16,1),low=0,high=10,device='cuda')
    # x1 = torch.tensor([1,2,3,4,5,0,0,0,0])
    # attention = LinearAttention(dim=512,embed_size=256,head=4).to('cuda')
    # print(attention(x,x,x))
    # coding = tokenizer.encode_batch(["Hello,what's your name ","hello,I don't know what's your name "])

    # block = TransformerBlock(dim_in=512,embed_dim=512,head=4,drop_rate=0.1).to(device='cuda')
    # enncoder = EncodeBlock(dim_in=512,embed_size=512,head=4,drop_rate=0.1,num=3).to('cuda')
    # trans = Transformer(dim_in=512,vocab_size=tokenizer.get_vocab_size(),head=4,num=3,embed_size=512).to('cuda')
    # out = trans(coding,coding)
    # print(out.shape)
    trainset = CustomDataset()
    trainer = Trainer(dim=512, vocab_size=trainset.tokenizer.get_vocab_size(), head=8, num=6, embed_size=512,
                      dataset=trainset, \
                      batch_size=128, config={'epoch': 200, 'device': 'mps'})
    trainer.train_loop()
    # encoding =tokenizer.encode("hello,what's your name")
    # src = {'attention_mask':torch.tensor(encoding.attention_mask),'ids':torch.tensor(encoding.ids)}
    # tar_enc = tokenizer.encode("<BOS>")
    # tar = {'attention_mask':torch.tensor(tar_enc.attention_mask).unsqueeze(0),'ids':torch.tensor(tar_enc.ids).unsqueeze(0)}
    # tar_out = trainer.inference(src=src,tar=tar)
    # tar_out.squeeze_(0)
    # tokenizer.decode(tar_out.tolist())
    # print(tar_out)
