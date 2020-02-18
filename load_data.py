# -*- coding: utf-8 -*-
import pandas as pd


# 读取txt文件
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = [_.strip() for _ in f.readlines()]

    labels, texts = [], []
    for line in content:
        parts = line.split()
        label, text = parts[0], ''.join(parts[1:])
        labels.append(label)
        texts.append(text)

    return labels, texts

# 获取训练数据和测试数据，格式为pandas的DataFrame
def get_train_test_pd():
    file_path = 'data/train.txt'
    labels, texts = read_txt_file(file_path)
    train_df = pd.DataFrame({'label': labels, 'text': texts})

    file_path = 'data/test.txt'
    labels, texts = read_txt_file(file_path)
    test_df = pd.DataFrame({'label': labels, 'text': texts})

    return train_df, test_df


if __name__ == '__main__':

    train_df, test_df = get_train_test_pd()
    print(train_df.head())
    print(test_df.head())

    train_df['text_len'] = train_df['text'].apply(lambda x: len(x))
    print(train_df.describe())

