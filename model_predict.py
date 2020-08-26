# -*- coding: utf-8 -*-
# 模型预测

import os, json
import numpy as np
from bert.extract_feature import BertVector
from keras.models import load_model
from att import Attention

# 加载训练效果最好的模型
model_dir = './models'
files = os.listdir(model_dir)
models_path = [os.path.join(model_dir, _) for _ in files]
best_model_path = sorted(models_path, key=lambda x: float(x.split('-')[-1].replace('.h5', '')), reverse=True)[0]
print(best_model_path)
model = load_model(best_model_path, custom_objects={"Attention": Attention})

# 示例语句及预处理
text1 = '唐怡莹#唐石霞#唐怡莹，姓他他拉氏，名为他他拉·怡莹，又名唐石霞，隶属于满洲镶红旗。'
text1 = '谢才萍#文斌#谢才萍的丈夫文斌利用其兄文强的影响，公开找渝中区公安分局原局长彭长健，要他关照老婆的“经'
text1 = '文强#文斌#谢才萍的丈夫文斌利用其兄文强的影响，公开找渝中区公安分局原局长彭长健，要他关照老婆的“经'
text1 = '文强#谢才萍#谢才萍的丈夫文斌利用其兄文强的影响，公开找渝中区公安分局原局长彭长健，要他关照老婆的“经'
text1 = '崔新琴#黄晓明#那时，老师崔新琴如此评价黄晓明：“没有灵性，就是一块漂亮的木头。”'
text1 = '秦天#秦先生#这个看来并不高深的游戏却让秦天和的爸爸秦先生大伤脑筋。'
text1 = '马清伟#马桂烽#说到早前传出与家族拥百亿财产的富豪马清伟和薛芷伦大儿子马桂烽（Justin）分手，她说：“有这个传闻的时候我经纪人已经代为回答，这是两个人的事不会回应，今天也是一样，多谢大家关心。'
text1 = '李克农#李伦#6月25日，澎湃新闻（www.thepaper.cn）从李伦将军亲友处获悉，开国上将李克农之子、解放军原总后勤部副部长李伦中将于2019年6月25日凌晨在北京逝世，享年92岁。'
text1 = '利孝和#陆雁群#利孝和的妻子是陆雁群，尊称「利孝和夫人」，是现任无线非执行董事，香港著名慈善家及利希慎家族成员'
text1 = '张少怀#费贞绫#家庭出生演艺世家的张菲，父亲为台湾综艺大哥张少怀，叔叔是费玉清，姑姑是费贞绫。'
text1 = '查济民#刘璧如#刘璧如女士在香港妇女界是位出类拔萃的人物，与其夫婿查济民先生一道经营中国染厂 。'
text1 = '申军良#申聪#3月7日，申军良夫妇与申聪认亲。'
text1 = '邓小平#邓文明#邓小平的父亲叫邓文明，是一个小地主。'
text1 = "波林#小艾伯特·阿诺德#这对波林尤其不易，因为生小艾伯特·阿诺德时，她已经36岁。"
# text1 = '陈发科#陈小旺#陈发科的爷爷是陈式太极拳一代宗师陈小旺，父亲是陈照旭。'
# text1 = '陈发科#陈照旭#陈发科的爷爷是陈式太极拳一代宗师陈小旺，父亲是陈照旭。'
# text1 = '何鸿燊#叶德利#叶德利的妻子是何鸿燊胞妹何婉婉'
text1 = '徐寿#华蘅芳#3年后，徐寿和华蘅芳同心协力，制造出了我国第一艘机动木质轮船，“长五十余尺，每一时能行四十余里”。'
per1, per2, doc = text1.split('#')
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
print('原文: %s' % text1)
print('预测人物关系: %s' % id_rel_dict[y])

# 预测分类结果为：夫妻
