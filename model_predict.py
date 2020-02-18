# -*- coding: utf-8 -*-
# 模型预测

import json
import numpy as np
from bert.extract_feature import BertVector
from keras.models import load_model
from att import Attention

# 加载模型
model = load_model('people_relation.h5', custom_objects={"Attention": Attention})

# 示例语句及预处理
text = '赵金闪#罗玉兄#在这里，赵金闪和罗玉兄夫妇已经生活了大半辈子。他们夫妇都是哈密市伊州区林业和草原局的护林员，扎根东天山脚下，守护着这片绿。'
per1, per2, doc = text.split('#')
text = '$'.join([per1, per2, doc.replace(per1, len(per1)*'#').replace(per2, len(per2)*'#')])
print(text)


# 利用BERT提取句子特征
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=80)
vec = bert_model.encode([text])["encodes"][0]
x_train = np.array([vec])

# 模型预测并输出预测结果
predicted = model.predict(x_train)
y = np.argmax(predicted[0])

with open('data/rel_dict.json', 'r', encoding='utf-8') as f:
    rel_dict = json.load(f)

id_rel_dict = {v:k for k,v in rel_dict.items()}
print(id_rel_dict[y])

# 预测分类结果为：夫妻
