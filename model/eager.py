import time
import math
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.utils import shuffle
from ConvKB.model.net import MyConvKB
import math
import numpy as np
from numpy.random import rand
from numpy import zeros
import oneflow as flow
from oneflow import nn
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import argparse
flow.enable_eager_execution()


#一般来说，为了平衡收敛速度和硬件需求，我们采用batch的形式进行训练。这里使用一个自己实现的next_batch函数实现从完整的训练/测试集中按序列取batch。
def next_batch(data, batch_size):
    (x,label)=data
    data_length = len(label)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield (x[start_index:end_index],label[start_index:end_index])

def ConvKB_loss(prediction,label):
    '''
    论文中value1没有取负
    '''
    value1 = flow.mul(label, prediction)
    value2=flow.exp(-value1)
    value3 = flow.add(value2, 1)
    value4=flow.log(value3)
    len=label.shape[0]
    lossSum=flow.sum(value4,dim=0)
    avgLoss = flow.div(lossSum, len)
    return avgLoss

def train(model, train_set, valid_set, device, num_epoch, batch_size, lr):
    train_len=len(train_set[1])
    print('train数据有%s条'%train_len)
    trainTimes_perEpoch=int(train_len/batch_size)
    print('一个epoch需要训练%s'%trainTimes_perEpoch)
    valid_len=len(valid_set[1])
    print('valid数据有%s条'%valid_len)


    model = model.to(device)

    loss_func = ConvKB_loss
    optimizer = flow.optim.SGD(model.parameters(), lr=lr)

    def pre_batch(batch):
        (x,label)=batch
        
        x = flow.tensor(x,dtype=flow.float32).to(device)  # (batch, seq_len)
        prediction = model(x)
        prediction=flow.reshape(prediction,(-1,1))
        label=flow.tensor(label,dtype=flow.float32).to(device)
        label=flow.reshape(label,(-1,1))
        return prediction, label

    train_loss_log = []
    train_acc_log=[]
    valid_loss_log = []
    valid_acc_log=[]
    trained_batches = 0
    for epoch in range(num_epoch):
        print('----------epoch:%s-------------'%epoch)
        for batch in next_batch(train_set, batch_size=batch_size):
            prediction, label = pre_batch(batch)

            _prediction=list(map(lambda p: 2*(p>0 or p==0)-1,prediction.detach().to('cpu').numpy()))
            train_acc=accuracy_score(_prediction,label.detach().to('cpu').numpy())
            train_acc_log.append(train_acc)

            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss=loss.detach().to('cpu').numpy()[0]
            train_loss_log.append(train_loss)
            trained_batches += 1

            if trained_batches % int(trainTimes_perEpoch) == 0:
                with flow.no_grad():
                    all_prediction = []
                    all_label=[]
                    for batch in next_batch(valid_set, batch_size=512):
                        prediction, label = pre_batch(batch)
                        all_prediction+=prediction.detach().to('cpu').numpy().flatten().tolist()
                        all_label+=label.detach().to('cpu').numpy().flatten().astype('int32').tolist()
                    #_all_prediction是映射到-1与1，用来计算acc的，all_prediction用来计算convkb_loss
                    _all_prediction=list(map(lambda p: 2*(p>0 or p==0)-1,all_prediction))
                    _all_prediction=np.array(_all_prediction)
                    all_label=np.array(all_label)
                    


                    valid_acc=accuracy_score(_all_prediction,all_label)
                    all_prediction=flow.tensor(all_prediction,dtype=flow.float32).to(device)
                    all_prediction=flow.reshape(all_prediction,(-1,1))
                    all_label=flow.tensor(all_label,dtype=flow.float32).to(device)
                    all_label=flow.reshape(all_label,(-1,1))
                    valid_loss=loss_func(all_prediction,all_label)
                    valid_loss=valid_loss.detach().to('cpu').numpy()
                    valid_acc_log.append(valid_acc)
                    valid_loss_log.append(valid_loss)
                    print('训练集：%s,acc: %.4f, convkb_loss: %.4f'%(trained_batches,train_acc_log[-1],train_loss_log[-1]))
                    print('验证集：%s,acc: %.4f, convkb_loss: %.4f' % (trained_batches,valid_acc_log[-1],valid_loss_log[-1]))
    return train_loss_log,train_acc_log,valid_loss_log,valid_acc_log

def plot_loss_metrics(score_log, loss_log):
    score_log = np.array(score_log)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.subplot(2, 2, 1)
    plt.plot(score_log[:, 0], c='#d28ad4')
    plt.ylabel('RMSE')

    plt.subplot(2, 2, 2)
    plt.plot(score_log[:, 1], c='#e765eb')
    plt.ylabel('MAE')

    plt.subplot(2, 2, 3)
    plt.plot(score_log[:, 2], c='#6b016d')
    plt.ylabel('MAPE')

    plt.subplot(2, 2, 4)
    plt.plot(loss_log)
    plt.ylabel('Loss')
    plt.xlabel('Number of batches')

    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddingDim', type=int, default=200)
    parser.add_argument('--kernelNum', type=int, default=100)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=9e-2)
    parser.add_argument('--dir', type=str, default='/home/shiyunxiao/ConvKB/data/Video/')
    parser.add_argument('--device', type=str, default='cuda:0')
    FLAGS = parser.parse_args()



    def transY(array=[]):
        y=[]
        for i in array:
            if i==0:
                y.append(-1)
            else:
                y.append(1)
        return np.array(y,dtype=float)
    #formFeedData()
    feedModelDic=np.load(FLAGS.dir+'feedModelDic.npy',allow_pickle=True)
    x_train_embedding= feedModelDic.item()['x_train_embedding']
    print('x_train_embedding.shape',x_train_embedding.shape)
    y_train= transY(feedModelDic.item()['y_train'])

    x_valid_embedding= feedModelDic.item()['x_valid_embedding']
    y_valid= transY(feedModelDic.item()['y_valid'])
    x_test_embedding= feedModelDic.item()['x_test_embedding']
    y_test= transY(feedModelDic.item()['y_test'])


    device=FLAGS.device
    my_convkb = MyConvKB(embeddingDim=FLAGS.embeddingDim,kernelNum=FLAGS.kernelNum)
    train_loss_log,train_acc_log,valid_loss_log,valid_acc_log = train(model=my_convkb,
                                    train_set=(x_train_embedding,y_train), valid_set=(x_valid_embedding,y_valid), device=device, 
                                    num_epoch=FLAGS.num_epoch, batch_size=FLAGS.batch_size, lr=FLAGS.lr)




    #model2(x_train_embedding.reshape(-1,100,3), y_train, x_valid_embedding.reshape(-1,100,3), y_valid,model1)
    # convCFKG_model(x_train_embedding.reshape(-1,100,3), y_train, x_valid_embedding.reshape(-1,100,3), y_valid)