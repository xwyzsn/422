import torch
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import re

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

from read_data import read_data


def remove_special_characters(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text


class CustomDataset(Dataset):

    def __init__(self, path="swaption2009/20k-en-zh-translation-pinyin-hsk", dir='./data'):

        self.tokenizer = Tokenizer.from_file('./tmp.json')
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<PAD>"), pad_token="<PAD>", length=16)
        self.tokenizer.enable_truncation(max_length=16)
        self.tokenizer.post_processor = TemplateProcessing(
            single="$A <EOS>",
            pair="<BOS> $A <EOS> $B:1 <EOS>:1",
            special_tokens=(
                ("<BOS>", tokenizer.token_to_id("<BOS>")),
                ("<EOS>", tokenizer.token_to_id("<EOS>")),
            )
        )
        self.data = read_data()
        self.data = self.process_data()

    def process_data(self):
        print("======= process ========")
        o = []
        for idx in range(len(self.data)):
            eng = remove_special_characters(self.data[idx]['eng']).strip()
            zh = remove_special_characters(self.data[idx]['zh']).strip()
            eng = self.tokenizer.encode(eng)
            zh = self.tokenizer.encode(zh)
            eng_ids = torch.tensor(eng.ids)
            eng_attention_mask = torch.tensor(eng.attention_mask)
            zh_ids = torch.tensor(zh.ids)
            zh_attention_maks = torch.tensor(zh.attention_mask)
            o.append({
                'eng':{'attention_mask':eng_attention_mask,'ids':eng_ids},
                'zh':{'attention_mask':zh_attention_maks,'ids':zh_ids}
            })
        print('===== finish ========')
        return o

    # def extract(self):
    #     print("====== extract ==========")
    #     o = []
    #     for idx in range(0, len(self.dataset), 5):
    #         eng = remove_special_characters(self.dataset[idx].get('text').split('english: ')[1].strip())
    #         zh = remove_special_characters(self.dataset[idx + 2].get('text').split('mandarin:')[1].strip())
    #         eng = self.tokenizer.encode(eng)
    #         zh = self.tokenizer.encode(zh)
    #         eng_ids = torch.tensor(eng.ids)
    #         eng_attention_mask = torch.tensor(eng.attention_mask)
    #         zh_ids = torch.tensor(zh.ids)
    #         zh_attention_mask = torch.tensor(zh.attention_mask)
    #         o.append({'eng': {'attention_mask': eng_attention_mask, 'ids': eng_ids},
    #                   'zh': {'attention_mask': zh_attention_mask, 'ids': zh_ids}})
    #     print("========== finish ==========")
    #     return o

    def __getitem__(self, index):
        return self.data[index]['eng'], self.data[index]['zh']

    def __len__(self):
        return len(self.data)


# tokenizer = Tokenizer(models.WordPiece())
# tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase(), normalizers.Strip(),
#                                              normalizers.BertNormalizer()])
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
#
# trainer = trainers.WordPieceTrainer(
#     vocab_size=30000,
#     special_tokens=["<PAD>", "<BOS>", "<EOS>", "[UNK]"]
# )
# path = "swaption2009/20k-en-zh-translation-pinyin-hsk"
# dir = './data'
# dataset = load_dataset(path=path, split='train',cache_dir=dir)


# def extract():
#     o = []
#     for idx in range(0, len(dataset), 5):
#         eng = dataset[idx].get('text')
#         zh = dataset[idx + 2].get('text')
#         o.append({'eng': eng, 'zh': zh})
#     return o


# data = read_data()
# training_data = [item['eng'] for item in data] + [item['zh'].replace('mandarin: ', '') for item in data]
# tokenizer.train_from_iterator(training_data, trainer=trainer)
tokenizer = Tokenizer.from_file('./tmp.json')
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<PAD>"),pad_token="<PAD>",length=16)
tokenizer.enable_truncation(max_length=16)
tokenizer.post_processor = TemplateProcessing(
    single="$A <EOS>",
    pair="<BOS> $A <EOS> $B:1 <EOS>:1",
    special_tokens=(
        ("<BOS>",tokenizer.token_to_id("<BOS>")),
        ("<EOS>",tokenizer.token_to_id("<EOS>")),
    )
)
# tokenizer.save('./tmp.json')
zhs = tokenizer.encode("你好，你叫什么名字")
print(zhs.ids)
print(tokenizer.decode(zhs.ids))
