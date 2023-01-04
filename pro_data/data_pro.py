# -*- coding: utf-8 -*-
import json
import pandas as pd
import re
import sys
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from operator import itemgetter
#import gensim
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

P_REVIEW = 0.85
MAX_DF = 0.7
MAX_VOCAB = 50000
DOC_LEN = 500
PRE_W2V_BIN_PATH = ""  # the pre-trained word2vec files


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def get_count(data, id):
    ids = set(data[id].tolist())
    return ids


def numerize(data):
    uid = list(map(lambda x: user2id[x], data['user_id']))
    iid = list(map(lambda x: item2id[x], data['item_id']))
    data['user_id'] = uid
    data['item_id'] = iid
    return data


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()


def bulid_vocbulary(xDict):
    rawReviews = []
    for (id, text) in xDict.items():
        rawReviews.append(' '.join(text))
    return rawReviews


def build_doc(u_reviews_dict, i_reviews_dict):
    '''
    1. extract the vocab
    2. fiter the reviews and documents of users and items
    '''
    u_reviews = []
    for ind in range(len(u_reviews_dict)):
        u_reviews.append(' <SEP> '.join(u_reviews_dict[ind]))

    i_reviews = []
    for ind in range(len(i_reviews_dict)):
        i_reviews.append('<SEP>'.join(i_reviews_dict[ind]))

    vectorizer = TfidfVectorizer(max_df=MAX_DF, max_features=MAX_VOCAB)
    vectorizer.fit(u_reviews)
    vocab = vectorizer.vocabulary_
    vocab[MAX_VOCAB] = '<SEP>'

    def clean_review(rDict):
        new_dict = {}
        for k, text in rDict.items():
            new_reviews = []
            for r in text:
                words = ' '.join([w for w in r.split() if w in vocab])
                new_reviews.append(words)
            new_dict[k] = new_reviews
        return new_dict

    def clean_doc(raw):
        new_raw = []
        for line in raw:
            review = [word for word in line.split() if word in vocab]
            if len(review) > DOC_LEN:
                review = review[:DOC_LEN]
            new_raw.append(review)
        return new_raw

    u_reviews_dict = clean_review(u_reviews_dict)
    i_reviews_dict = clean_review(i_reviews_dict)

    u_doc = clean_doc(u_reviews)
    i_doc = clean_doc(i_reviews)

    return vocab, u_doc, i_doc, u_reviews_dict, i_reviews_dict


def countNum(xDict):
    minNum = 100
    maxNum = 0
    sumNum = 0
    maxSent = 0
    minSent = 3000
    # pSentLen = 0
    ReviewLenList = []
    SentLenList = []
    for (i, text) in xDict.items():
        sumNum = sumNum + len(text)
        if len(text) < minNum:
            minNum = len(text)
        if len(text) > maxNum:
            maxNum = len(text)
        ReviewLenList.append(len(text))
        for sent in text:
            # SentLenList.append(len(sent))
            if sent != "":
                wordTokens = sent.split()
            if len(wordTokens) > maxSent:
                maxSent = len(wordTokens)
            if len(wordTokens) < minSent:
                minSent = len(wordTokens)
            SentLenList.append(len(wordTokens))
    averageNum = sumNum // (len(xDict))

    x = np.sort(SentLenList)
    xLen = len(x)
    pSentLen = x[int(P_REVIEW * xLen) - 1]
    x = np.sort(ReviewLenList)
    xLen = len(x)
    pReviewLen = x[int(P_REVIEW * xLen) - 1]

    return minNum, maxNum, averageNum, maxSent, minSent, pReviewLen, pSentLen


if __name__ == '__main__':

    start_time = time.time()
    assert (len(sys.argv) >= 2)
    filename = sys.argv[1]

    yelp_data = False
    if len(sys.argv) > 2 and sys.argv[2] == 'yelp':
        # yelp dataset
        yelp_data = True
        save_folder = '../dataset/' + filename[:-3] + "_data"
    else:
        # amazon dataset
        save_folder = '../dataset/' + filename[:-7] + "_data"
    print(f"数据集名称：{save_folder}")

    # if not os.path.exists(save_folder + '/train'):
    #     os.makedirs(save_folder + '/train')
    # if not os.path.exists(save_folder + '/val'):
    #     os.makedirs(save_folder + '/val')
    # if not os.path.exists(save_folder + '/test'):
    #     os.makedirs(save_folder + '/test')

    if len(PRE_W2V_BIN_PATH) == 0:
        print("Warning: the word embedding file is not provided, will be initialized randomly")
    file = open(filename, errors='ignore')

    print(f"{now()}: Step1: loading raw review datasets...")

    users_id = []
    items_id = []
    ratings = []
    reviews = []

    if yelp_data:
        for line in file:
            value = line.split('\t')
            reviews.append(value[2])
            users_id.append(value[0])
            items_id.append(value[1])
            ratings.append(value[3])
    else:
        for line in file:
            js = json.loads(line)
            if str(js['reviewerID']) == 'unknown':
                print("unknown user id")
                continue
            if str(js['asin']) == 'unknown':
                print("unkown item id")
                continue
            reviews.append(js['reviewText'])
            users_id.append(str(js['reviewerID']))
            items_id.append(str(js['asin']))
            ratings.append(str(js['overall']))

    data_frame = {'user_id': pd.Series(users_id), 'item_id': pd.Series(items_id),
                  'ratings': pd.Series(ratings)}
    data = pd.DataFrame(data_frame)  # [['user_id', 'item_id', 'ratings', 'reviews']]
    del users_id, items_id, ratings, reviews

    uidList, iidList = get_count(data, 'user_id'), get_count(data, 'item_id')
    userNum_all = len(uidList)
    itemNum_all = len(iidList)
    print("===============Start:all  rawData size======================")
    print(f"dataNum: {data.shape[0]}")
    print(f"userNum: {userNum_all}")
    print(f"itemNum: {itemNum_all}")
    print(f"data densiy: {data.shape[0] / float(userNum_all * itemNum_all):.4f}")
    print("===============End: rawData size========================")

    user2id = dict((uid, i) for (i, uid) in enumerate(uidList))
    item2id = dict((iid, i) for (i, iid) in enumerate(iidList))
    data = numerize(data)
    # 保存成CSV格式
    data.to_csv('../data/'+filename[:-5]+".data", header= None, index=False)

