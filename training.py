from dataset import mydata, Dataset
from model import myModel
from config import my_config
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_epoch(model, train_iterator, dev_iterator, optimzer, loss_fn):  # 训练模型
    '''
    :param model:模型
    :param train_iterator:训练数据的迭代器
    :param dev_iterator: 验证数据的迭代器
    :param optimzer: 优化器
    :param loss_fn: 损失函数
    '''
    
    model.train()

    losses = []
    for i, batch in enumerate(train_iterator):
        input_data = batch[0].float().to(device)
        label = batch[1].float().to(device)
        
        # print(input.shape)

        optimzer.zero_grad()

        pred = model(input_data)  # 预测

        pred = pred.view(-1)

        # print(pred)
        # print(label)

        loss = loss_fn(pred, label)  # 计算损失值

        loss.backward()  # 误差反向传播
        losses.append(loss.data.cpu().numpy())  # 记录误差
        optimzer.step()  # 优化一次

        # 打印batch级别日志
        print(("[step = %d] loss: %.3f ") % (i + 1, loss))


def evaluate_model(model, dev_iterator):  # 评价模型
    '''
    :param model:模型
    :param dev_iterator:待评价的数据
    :return:评价（准确率）
    '''
    model.eval()
    all_pred = []
    all_y = []
    for i, batch in enumerate(dev_iterator):
        input_data = batch[0].float().to(device)
        label = batch[1].float().to(device)

        y_pred = model(input_data)  # 预测
        # print(y_pred)
        all_pred.extend(y_pred.detach().cpu().numpy())
        temp = []
        for slabel in label.cpu().numpy():
            temp.append(slabel)
        all_y.extend(np.array(temp))
    all_y = np.array(all_y).flatten()
    all_pred = np.array(all_pred).flatten()
    return all_y, all_pred

if __name__ == '__main__':
    config = my_config()  # 配置对象实例化
    data_class = mydata(config)  # 数据类实例化
    dataset = Dataset(data_class, config)  # 数据预处理实例化

    dataset.load_data()  # 进行数据预处理

    train_iterator = dataset.train_iterator  # 得到处理好的数据迭代器
    dev_iterator = dataset.dev_iterator
    test_iterator = dataset.test_iterator


    # 初始化模型
    model = myModel(config)
    # todo cuda并行
    """
    if torch.cuda.device_count() > 3:  # 检查电脑是否有多块GPU
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)  # 将模型对象转变为多GPU并行运算的模型
    """
    model = model.to(device)
    
    print(summary(model, (9,9)))

    optimzer = torch.optim.Adam(model.parameters(), lr=config.lr)  # 优化器
    # loss_fn = nn.L1Loss(reduction='mean')
    loss_fn = nn.MSELoss()
    
    #print(model)

    torch.manual_seed(config.seeds)
    torch.cuda.manual_seed_all(config.seeds)

    y = []

    for i in range(config.epoch):
        print(f'epoch:{i + 1}')
        run_epoch(model, train_iterator, dev_iterator, optimzer, loss_fn)

        # 训练一次后评估一下模型
        yt, yp = evaluate_model(model, dev_iterator)
        
        #print(yt)
        #print(yp)
        score = 0
        for j in range(len(yt)):
            score += abs(yt[j] - yp[j])
        y.append(score)
        
        print("该epoch的损失函数值为:" + str(score))
        
        # 保存模型
    torch.save(model, "./model/BP_jerk.pth") 


    # 训练完画图
    x = [i for i in range(1,len(y) + 1)]
    fig = plt.figure()
    plt.plot(x, y, marker='o', markersize=4)
    plt.xticks(x)
    plt.savefig("./result/MSEloss_jerk.png")