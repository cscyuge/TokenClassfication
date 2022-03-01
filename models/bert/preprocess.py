import pickle
import torch
import numpy as np
from tqdm import tqdm
import re
from transformers import BertTokenizer, BertModel
from keras.preprocessing.sequence import pad_sequences
import os
import asyncio
from pprint import pprint
import argparse
import sys
sys.path.append("../../")
from utils.kmp import KMP
import numpy as np
import json
kmp = KMP()

CLS = '[CLS]'
SEP = '[SEP]'

bert_model = 'hfl/chinese-bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(bert_model)
# model = BertModel.from_pretrained(bert_model)
train_keys = pickle.load(open('../../data/train_keys.pkl','rb'))
L = len(tokenizer)
print(L)
new_words = []
for keyword in tqdm(train_keys):
    if keyword=='':
        print('empty!')
        continue
    tokenizer.add_tokens([keyword])
    if len(tokenizer)>L:
        L = len(tokenizer)
        new_words.append(keyword)

# model.resize_token_embeddings(len(tokenizer))
# for i, keyword in tqdm(enumerate(new_words)):
#     model.embeddings.word_embeddings.weight[-len(new_words)+i, :] = keywords_all[keyword]
print(len(tokenizer))
print(tokenizer.tokenize('白癜风得到的依从性关于其他的痛风石'))


def preprocess(dataset_all, path_to, index):
    global keywords_all
    src_all, src_ids, src_tokens = [], [], []
    tar_masks = []
    keywordset_list = []
    worker_num = 8
    dataset = []
    for i in range(worker_num):
        if i == index:
            for j, data in enumerate(
                    dataset_all[
                    i * len(dataset_all) // worker_num: (i + 1) * len(dataset_all) // worker_num]):
                dataset.append((i * len(dataset_all) // worker_num + j, data))

    for data in tqdm(dataset, ascii=True, ncols=50):
        contents = data[1]['contents']
        tokens = [CLS]
        masks = [0]
        # tokens = [CLS] + tokenizer.tokenize(re.sub('\*\*', '', src).lower()) + [SEP]
        _keywords = []

        for content in contents:
            src = content['text']
            _src_tokens = tokenizer.tokenize(re.sub('\*\*', '', src).lower())
            src_masks = np.array([0 for _ in range(len(_src_tokens))])
            for tooltip in content['tooltips']:
                key = tooltip['origin']
                keyword_tokens = tokenizer.tokenize(key)
                i = kmp.kmp(_src_tokens, keyword_tokens)
                l = len(keyword_tokens)
                if i != -1:
                    src_masks[i] = 1
                    if l >= 2:
                        print("get keyword token len >= 2")
                        src_masks[i + l - 1] = 4
                        src_masks[i + 1:i + l - 1] = 3
                    _keywords.append(key)
            tokens += _src_tokens
            masks += list(src_masks)

        tokens += [SEP]
        masks += [0]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        src_ids.append(ids)
        tar_masks.append(masks)
        keywordset_list.append(_keywords)
        src_tokens.append(tokens)

    print(len(src_ids))
    src_ids_smaller, tar_masks_smaller,keywords_smaller,src_tokens_smaller = [], [], [], []
    max_len = 512
    indexs = []
    for i,(src, masks,keywords, tokens) in enumerate(zip(src_ids, tar_masks,keywordset_list, src_tokens)):
        if len(src) < max_len and len(src) > 2:
            src_ids_smaller.append(src)
            tar_masks_smaller.append(masks)
            keywords_smaller.append(keywords)
            indexs.append(i)
            src_tokens_smaller.append(tokens)

    src_ids, tar_masks, keywordset_list, src_tokens = src_ids_smaller, tar_masks_smaller, keywords_smaller,src_tokens_smaller
    print(len(src_ids))

    tag_values = [0, 1, 2]
    tag2idx = {t: i for i, t in enumerate(tag_values)}

    src_ids = pad_sequences(src_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    tar_masks = pad_sequences(tar_masks, maxlen=max_len, dtype="long", value=tag2idx[2], truncating="post", padding="post")

    src_masks = [[float(i != 0.0) for i in ii] for ii in src_ids]

    with open(os.path.join(path_to, 'src_ids_{}.pkl'.format(index)), 'wb') as f:
        pickle.dump(src_ids, f)
    with open(os.path.join(path_to, 'src_masks_{}.pkl'.format(index)), 'wb') as f:
        pickle.dump(src_masks, f)
    with open(os.path.join(path_to, 'tar_masks_{}.pkl'.format(index)), 'wb') as f:
        pickle.dump(tar_masks, f)
    with open(os.path.join(path_to, 'keywordset_list_{}.pkl'.format(index)), 'wb') as f:
        pickle.dump(keywordset_list, f)

    with open(os.path.join(path_to,'data_{}.txt'.format(index)),'w', encoding='utf-8') as f:
        for src, masks in zip(src_tokens, tar_masks):
            for token, mask in zip(src, masks):
                f.write(token+' '+str(mask)+'\n')
            f.write('\n')

    # for ids, masks in zip(src_ids[:5], tar_masks[:5]):
    #     tokens = tokenizer.convert_ids_to_tokens(ids)
    #     for token, mask in zip(tokens, masks):
    #         print(token, mask)

def main(index):
    dataset = json.load(open('../../data/dataset_new_2.json', 'r', encoding='utf-8'))
    total = len(dataset)
    print('train dataset:')
    preprocess(dataset[:int(total/10*8)], './data/train',index)
    print('test dataset:')
    preprocess(dataset[int(total/10*8):int(total/10*9)], './data/test',index)
    print('valid dataset:')
    preprocess(dataset[int(total/10*9):], './data/valid',index)
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int)
    args = parser.parse_args()
    print(args.i)
    main(args.i)



