#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random
from tqdm import tqdm
import numpy as np
from ConvKB.util.tool import dataProcess
class Recommend():

    def __init__(self):
        self.df = dataProcess()
        self.df.context='/home/shiyunxiao/ConvKB/data/Beauty/'
        self.df.loadDicTypeConstrain()

    def topN(self,predictResult, n, trueIndex):
        '''

        :param predictResult: 预测概率列表
        :param n: topN,N=?
        :param productsIds: 用户真实购买列表
        :param trueIndex: 为0
        :return:
        '''
        # 距离前n小
        result_label = np.argpartition(predictResult, n)[:n]
        if trueIndex in result_label:
            return 1
        else:
            return 0

    def TransE(self,x_valid):
        # 欧氏距离
        def eucliDist(A, B):
            return np.sqrt(sum(np.power((A - B), 2)))

        List = []
        for triple in x_valid:
            headEmbedding = triple[0]
            tailEmbedding = triple[1]
            relationEmbedding = triple[2]
            sim = eucliDist(headEmbedding + relationEmbedding, tailEmbedding)
            List.append(sim)
        return List

    def caculateMR(self,disctanceList=[],appendSign=0):
        validDistance=disctanceList[appendSign]
        closerNum=0
        for i in disctanceList:
            if i<validDistance:
                closerNum+=1
        return closerNum+1
    def getMR(self,headName='',tailName='',relationName=''):
        headEmbedding = self.df.getEmbeddingByName(name=headName,type='entity')
        tailEmbedding = self.df.getEmbeddingByName(name=tailName,type='entity')
        relationEmbedding = self.df.getEmbeddingByName(name=relationName,type='relation')
        x_valid = []
        allRevelantTails=self.df.dicTypeConstrain[relationName]['tailNameList'][:]
        try: #怕玩意里面有
            allRevelantTails.remove(tailName)
        except:
            pass
        for otherTail in allRevelantTails:
            otherTailEmbedding = self.df.getEmbeddingByName(name=otherTail, type='entity')
            x_valid.append([headEmbedding,otherTailEmbedding,relationEmbedding])
        # 再把正确的三元组加入
        try:
            appendSign = random.randint(0, len(x_valid) - 1)
        except:
            return 0,0
        x_valid.insert(appendSign, [headEmbedding, tailEmbedding, relationEmbedding])
        x_valid = np.array(x_valid)
        disctance = self.TransE(x_valid)  # 这里是距离，越小越好
        rank=self.caculateMR(disctance,appendSign)
        return rank,len(disctance)

    def getHit(self,headName='',tailName='',relationName=''):
        testSize=100

        headEmbedding = self.df.getEmbeddingByName(name=headName,type='entity')
        tailEmbedding = self.df.getEmbeddingByName(name=tailName,type='entity')
        relationEmbedding = self.df.getEmbeddingByName(name=relationName,type='relation')
        x_valid = []
        while len(x_valid) is not testSize-1:
            # 生成被破坏的三元组,改变的是tail
            invalidTripleEmbedding = self.df.generateInvalidTriple(headName=headName, tailName=tailName,
                                                                   relationName=relationName)
            x_valid.append(invalidTripleEmbedding)
        # 再把正确的三元组加入
        appendSign = random.randint(0, testSize - 1)
        x_valid.insert(appendSign, [headEmbedding, tailEmbedding, relationEmbedding])
        x_valid=np.array(x_valid)
        buyProbability=self.TransE(x_valid) #这里的概率是相似度，相似度越大，认为概率就越大
        # hit==1表示topN中有，hit==1表示没有

        hit1 = self.topN(buyProbability, 1, appendSign)
        hit3 = self.topN(buyProbability, 3, appendSign)
        hit10 = self.topN(buyProbability, 10, appendSign)

        return hit1,hit3,hit10

    def caculateHitRation_ifStr(self,theRelation=''):
        '''
        用于形成convCFKG的验证数据
        格式为
        :return:
        数据格式为字典：
         '94474': ['4033', '8435', '22442'],
         key为user实体，value是product实体list
        '''

        times = 0
        hitNum_1 = 0
        hitNum_3 = 0
        hitNum_10 = 0
        with open('/home/shiyunxiao/ConvKB/data/Beauty/valid.txt') as f:

            for line in f.readlines():
                array = line.strip().split()
                headName = array[0]
                tailName = array[2]
                relationName = array[1]
                if relationName !=theRelation:
                    continue

                hit_num1, hit_num3, hit_num10 = self.getHit(headName, tailName, relationName)
                hitNum_1 += hit_num1
                hitNum_3 += hit_num3
                hitNum_10 += hit_num10
                times += 1
                if times%4000==0:
                    break

            print('-' * 20)
            print('relation: %s'%theRelation)
            print('productNum:%s' % times)
            print('hit@1:%s' % (hitNum_1 / times))
            print('hit@3:%s' % (hitNum_3 / times))
            print('hit@10:%s' % (hitNum_10 / times))
            print('-' * 20)
            # break
    def caculateMRandMRR_ifStr(self,theRelation=''):
        '''
        用于形成convCFKG的验证数据
        格式为
        :return:
        数据格式为字典：
         '94474': ['4033', '8435', '22442'],
         key为user实体，value是product实体list
        '''
        times = 0
        RANK = 0
        R_RANK = 0
        SETSIZE = 0
        with open('/home/shiyunxiao/ConvKB/data/Beauty/valid.txt') as f:

            for line in f.readlines():
                array = line.strip().split()
                headName = array[0]
                tailName = array[2]
                relationName = array[1]
                if relationName != theRelation:
                    continue
                rank, setSize = self.getMR(headName, tailName, relationName)
                print('-----%s   %s------' % (rank, setSize))
                if rank == 0 and setSize == 0:
                    continue
                SETSIZE += setSize
                RANK += rank
                R_RANK += 1 / rank
                times += 1
            print('-' * 20)
            print('relation: %s'%theRelation)
            print('productNum:%s' % times)
            print('MR:%s' % int(RANK / times))
            print('MRR:%s' % (R_RANK / times))
            print('MsetSize:%s' % (SETSIZE / times))
            print('-' * 20)
            # break
if __name__ == '__main__':
    rm=Recommend()
    #rm.caculateHitRation_ifStr('/media_common/netflix_genre/titles')
    relationList=rm.df.getAllRelationNames()
    for relation in tqdm(relationList):
        rm.caculateHitRation_ifStr(relation)
