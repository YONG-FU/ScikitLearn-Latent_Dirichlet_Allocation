# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import jieba

jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('京州', True)
# 第一个文档分词#
with open("nlp_test0.txt", encoding="utf-8") as f:
    document1 = f.read()

    document1_cut = jieba.cut(document1)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document1_cut)
    with open('nlp_test1.txt', 'w', encoding="utf-8") as f1:
        f1.write(result)

# 第二个文档分词#
with open('nlp_test2.txt', encoding="utf-8") as f:
    document2 = f.read()

    document2_cut = jieba.cut(document2)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document2_cut)
    with open('nlp_test3.txt', 'w', encoding="utf-8") as f2:
        f2.write(result)

# 第三个文档分词#
with open('nlp_test4.txt', encoding="utf-8") as f:
    document3 = f.read()

    document3_cut = jieba.cut(document3)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document3_cut)
    with open('nlp_test5.txt', 'w', encoding="utf-8") as f3:
        f3.write(result)

with open('nlp_test1.txt', encoding="utf-8") as f4:
    result1 = f4.read()
print(result1)
with open('nlp_test3.txt', encoding="utf-8") as f5:
    result2 = f5.read()
print(result2)
with open('nlp_test5.txt', encoding="utf-8") as f6:
    result3 = f6.read()
print(result3)

#从文件导入停用词表
with open("stop_words.txt", encoding="utf-8") as stop_words_dictionary:
    stop_words_content = stop_words_dictionary.read()
    stop_word_list = stop_words_content.splitlines()

corpus = [result1, result2, result3]
count_vector = CountVectorizer(stop_words=stop_word_list)
count_term_frequency = count_vector.fit_transform(corpus)
print(count_term_frequency)

latent_dirichlet_allocation = LatentDirichletAllocation(n_components=2, random_state=0)
document_result = latent_dirichlet_allocation.fit_transform(count_term_frequency)

print(document_result)
print(latent_dirichlet_allocation.components_)