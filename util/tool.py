
import random
from tqdm import tqdm
import numpy as np
import os

class dataProcess:
    context='/home/shiyunxiao/ConvKB/data/Beauty/'
    dicTypeConstrain={}
    def __init__(self,path='/home/shiyunxiao/ConvKB/data/Beauty/'):
        self.context= path
        self.buildDict()

    def loadDicTypeConstrain(self):
        print('---------------------------')
        self.dicTypeConstrain = np.load(self.context + 'dicTypeConstrain.npy', allow_pickle=True).item()
        print(list(self.dicTypeConstrain.keys()))



    def buildDict(self):
        '''
        建立字典
        self.entity_id_name_dict, 实体的，id2name
        self.entity_name_id_dict={},实体的，name2id
        self.entity_name_embedding={},实体的，name2embedding
        self.relation_id_name_dict={}, 关系的，id2name
        self.relation_name_id_dict={},关系的，name2id
        self.relation_name_embedding={},关系的，name2embedding
        '''
        context=self.context
        entity2idfileName='entity2id.txt'
        entityEmbedding_path='ent_embeddings.npy'
        relation2idfileName='relation2id.txt'
        relationEmbedding_path='rel_embeddings.npy'
        entityEmbeddingArray=np.load(context+entityEmbedding_path)
        relationEmbeddingArray=np.load(context+relationEmbedding_path)
        entityEmbedding_dim=entityEmbeddingArray.shape[1]
        relationEmbedding_dim=relationEmbeddingArray.shape[1]
        if entityEmbedding_dim==relationEmbedding_dim:
            self.embed_dim=entityEmbedding_dim
            print('实体与关系的embedding维度为:',self.embed_dim)
        else:
            print('实体与关系的embedding维度不一致')
        self.entity_id_name_dict={}
        self.entity_name_id_dict={}
        self.entity_name_embedding={}
        #print('字典建立中！')
        with open(context+entity2idfileName) as file1:
            for line in file1.readlines()[1:]:
                array = line.strip().split('\t')
                entityName=array[0]
                entityId=int(array[1])
                entityEmbedding=entityEmbeddingArray[entityId]
                self.entity_id_name_dict[entityId]=entityName
                self.entity_name_id_dict[entityName]=entityId
                self.entity_name_embedding[entityName]=entityEmbedding

        self.relation_id_name_dict={}
        self.relation_name_id_dict={}
        self.relation_name_embedding={}
        with open(context+relation2idfileName) as file2:
            for line in file2.readlines()[1:]:
                array = line.strip().split('\t')
                relationName=array[0]
                relationId=int(array[1])
                relationEmbedding=relationEmbeddingArray[relationId]
                self.relation_id_name_dict[relationId]=relationName
                self.relation_name_id_dict[relationName]=relationId
                self.relation_name_embedding[relationName]=relationEmbedding

        #print('字典建立完成！')
    def getAllRelationNames(self):
        '''
        以list返回所有relation的name
        '''
        return list(self.relation_name_id_dict.keys())

    def prepare_dicTypeConstrain(self):
        '''
        初始化一次，生成npy文件
        从train中形成relation约束,dic dic list类型
        dic[relationName]={'headNameList':[],'tailNameList':[]}
        headNameList为该relation下，所有的头实体列表；
        tailNameList为该relation下，所有的尾实体列表；
        :return:
        '''
        context=self.context

        #先查阅有没有,有的话直接读文件
        if os.path.exists(context+'dicTypeConstrain.npy'):
            self.loadDicTypeConstrain()
            return 

        fileName='train.txt'
        dic={}
        with open(context+fileName) as f:
            for line in tqdm(f.readlines()):
                array=line.strip().split('\t')
                headName=array[0]
                tailName=array[2]
                relationName=array[1]
                #如果是第一次遇到新的关系，初始化
                if relationName not in dic.keys():
                    dic[relationName]={'headNameList':[],
                                       'tailNameList':[]}
                #那么就是关系已经在里面了，可以用键访问
                else:
                    #如果headName,tailName不在list，加入；如果在，就不用再加了
                    if headName not in dic[relationName]['headNameList']:
                        dic[relationName]['headNameList'].append(headName)
                    if tailName not in dic[relationName]['tailNameList']:
                        dic[relationName]['tailNameList'].append(tailName)
        dicArray=np.array(dic)
        np.save(context+'dicTypeConstrain.npy',dicArray)
        self.loadDicTypeConstrain()

    def getEmbeddingByName(self,name='',type=''):
        vec = []
        if type=='entity':
            vec=self.entity_name_embedding[name]
        elif type=='relation':
            vec=self.relation_name_embedding[name]
        return vec
    def getIdByName(self,name='',type=''):
        id=None
        if type=='entity':
            id=self.entity_name_id_dict[name]
        elif type=='relation':
            id=self.relation_name_id_dict[name]
        return id
    def generateInvalidTriple(self,headName='',tailName='',relationName=''):
        tailNameList=self.dicTypeConstrain[relationName]['tailNameList'][:]
        if tailName in tailNameList:
            tailNameList.remove(tailName)


        #如果只有一个候选的tail(已经移除)，那么返回0000
        if len(tailNameList) ==0 :
            headEmbedding = np.zeros((1,self.embed_dim))[0]
            tailEmbedding = np.zeros((1,self.embed_dim))[0]
            relationEmbedding = np.zeros((1,self.embed_dim))[0]
            return [headEmbedding, tailEmbedding, relationEmbedding]
        else:

            randomNum=random.randint(0, len(tailNameList) - 1)
            randomTailName=tailNameList[randomNum]

            headEmbedding = self.getEmbeddingByName(name=headName, type='entity')
            tailEmbedding = self.getEmbeddingByName(name=randomTailName, type='entity')
            relationEmbedding = self.getEmbeddingByName(name=relationName, type='relation')

            return [headEmbedding,tailEmbedding,relationEmbedding]

    def read_train_valid_test(self):
        '''
        Train_valid_test 分隔符为\t headName  relationName tailName
        :return:
        '''
        fileNameList=['valid.txt','train.txt','test.txt']
        context=self.context
        feedModelDic={}

        for fileName in fileNameList:

            x = []
            y = []
            with open(context+fileName) as f:
                times = 0
                for line in tqdm(f.readlines()):
                    array=line.strip().split('\t')
                    headName=array[0]
                    tailName=array[2]
                    relationName=array[1]
                    headEmbedding=self.getEmbeddingByName(name=headName,type='entity')
                    tailEmbedding = self.getEmbeddingByName(name=tailName, type='entity')
                    relationEmbedding = self.getEmbeddingByName(name=relationName, type='relation')

                    #有效的三元组
                    x.append([headEmbedding,tailEmbedding,relationEmbedding])
                    y.append(1)

                    #生成被破坏的三元组,改变的是tail
                    invalidTripleEmbedding=self.generateInvalidTriple(headName=headName,tailName=tailName,relationName=relationName)
                    x.append(invalidTripleEmbedding)
                    y.append(0)
                    if times % 2000 == 0:
                        print('%s已经%s次' % (fileName, times))
                    times += 1
            x=np.array(x)
            y=np.array(y)
            feedModelDic['x_%s_embedding'%fileName[:-4]]=x
            feedModelDic['y_%s'%fileName[:-4]]=y

        feedModelDic=np.array(feedModelDic)
        np.save(self.context+'feedModelDic.npy',feedModelDic)


if __name__ == '__main__':
    dp=dataProcess(path='/home/shiyunxiao/ConvKB/data/Video/')
    dp.buildDict()
    dp.prepare_dicTypeConstrain()
    dp.read_train_valid_test()
 
