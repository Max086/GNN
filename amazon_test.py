from torch_geometric.data import Dataset, Data
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric
from torch_geometric.data import Data
import graphviz
from graphviz import Source, Digraph
import networkx as nx
import pydot
import os
from gensim.models import Word2Vec
from torch.nn import LSTM
import pandas as pd
import pickle
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch.nn import Linear
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.data import DataLoader
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR
import matplotlib as mpl

# 删除掉上次运行得到的权重文件
path = "./dataset/processed"


def pt_remove():
    if len(os.listdir(path)):
        for file in os.listdir(path):
            file_path = path + "/" + file
            os.remove(file_path)


# 权重文件删除
pt_remove()


class MyData(Data):
    def __init__(
        self,
        x=None,
        edge_index=None,
        edge_attr=None,
        y=None,
        pos=None,
        edge_index_IVI=None,
        edge_index_IBI=None,
        edge_index_ITI=None,
        edge_index_IOI=None,
        **kwargs,
    ):
        super().__init__(x, edge_index, edge_attr, y, pos)
        self.edge_index_IVI = edge_index_IVI
        self.edge_index_IBI = edge_index_IBI
        self.edge_index_ITI = edge_index_ITI
        self.edge_index_IOI = edge_index_IOI


class MyDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyDataset, self).__init__(root, transform, pre_transform)

    # 原始文件位置
    @property
    def raw_file_names(self):
        return ["amazon.pkl"]

    # 文件保存位置
    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        pass

    # 数据处理逻辑
    # def process(self):
    #     path = "./dataset/amazon.pkl"
    #     f = open(path, "rb")
    #     amazon = pickle.load(f)
    #
    #     # 缩放20倍
    #     e_from_IVI = []
    #     e_end_IVI = []
    #     for i in range(len(amazon["IVI"])):
    #         # ids = np.nonzero(amazon["IVI"][i])[0].tolist()
    #         for j in range(len(ids)):
    #             if j % 20 != 0:
    #                 amazon["IVI"][i][ids[j]] = 0
    #             else:
    #                 if i != ids[j]:
    #                     e_from_IVI.append(i)
    #                     e_end_IVI.append(ids[j])
    #     edge_index_IVI = torch.tensor([e_from_IVI, e_end_IVI], dtype=torch.long)
    #     print(len(edge_index_IVI[0]))
    #
    #     e_from_IBI = []
    #     e_end_IBI = []
    #     for i in range(len(amazon["IBI"])):
    #         ids = np.nonzero(amazon["IBI"][i])[0].tolist()
    #         for j in range(len(ids)):
    #             if j % 20 != 0:
    #                 amazon["IBI"][i][ids[j]] = 0
    #             else:
    #                 if i != ids[j]:
    #                     e_from_IBI.append(i)
    #                     e_end_IBI.append(ids[j])
    #     edge_index_IBI = torch.tensor([e_from_IBI, e_end_IBI], dtype=torch.long)
    #     print(len(edge_index_IBI[0]))
    #
    #     e_from_ITI = []
    #     e_end_ITI = []
    #     for i in range(len(amazon["ITI"])):
    #         ids = np.nonzero(amazon["ITI"][i])[0].tolist()
    #         for j in range(len(ids)):
    #             if j % 20 != 0:
    #                 amazon["ITI"][i][ids[j]] = 0
    #             else:
    #                 if i != ids[j]:
    #                     e_from_ITI.append(i)
    #                     e_end_ITI.append(ids[j])
    #     edge_index_ITI = torch.tensor([e_from_ITI, e_end_ITI], dtype=torch.long)
    #     print(len(edge_index_ITI[0]))
    #
    #     e_from_IOI = []
    #     e_end_IOI = []
    #     for i in range(len(amazon["IOI"])):
    #         if i % 20 == 0:
    #             for j in np.nonzero(amazon["IOI"][i])[0]:
    #                 if i != j:
    #                     e_from_IOI.append(i)
    #                     e_end_IOI.append(j)
    #     edge_index_IOI = torch.tensor([e_from_IOI, e_end_IOI], dtype=torch.long)
    #     print(len(edge_index_IOI[0]))
    #
    #     y = torch.tensor(amazon["label"], dtype=torch.float)
    #     print(y)
    #
    #     #         x = []
    #     #         for i in data['train_idx'][0]:
    #     #             x.append(data['feature'][i])
    #     x = torch.tensor(amazon["feature"], dtype=torch.float)
    #     #         x = torch.tensor(np.array(x), dtype=torch.float)
    #     print(x)
    #
    #     edge_index = torch.tensor([[0], [0]])
    #     edge_attr = torch.tensor([0])
    #
    #     data = MyData(
    #         x=x,
    #         y=y,
    #         edge_index=edge_index,
    #         edge_attr=edge_attr,
    #         edge_index_IVI=edge_index_IVI,
    #         edge_index_IBI=edge_index_IBI,
    #         edge_index_ITI=edge_index_ITI,
    #         edge_index_IOI=edge_index_IOI,
    #     )
    #
    #     torch.save(data, os.path.join(self.processed_dir, "data.pt"))
    def process(self):
        path = "D:\\PycharmProjects\\pythonProject2\\pytorch_HAN-master\\data\\amazon.pkl"
        f = open(path, 'rb')
        data = pickle.load(f)
        e_from_IVI = []
        e_end_IVI = []
        for i in range(len(data['IVI'])):
            for j in np.nonzero(data['IVI'][i])[0]:
                if i != j:
                    e_from_IVI.append(i)
                    e_end_IVI.append(j)
        edge_index_IVI = torch.tensor([e_from_IVI, e_end_IVI], dtype=torch.long)
        print(edge_index_IVI)

        e_from_IBI = []
        e_end_IBI = []
        for i in range(len(data['IBI'])):
            for j in np.nonzero(data['IBI'][i])[0]:
                if i != j:
                    e_from_IBI.append(i)
                    e_end_IBI.append(j)
        edge_index_IBI = torch.tensor([e_from_IBI, e_end_IBI], dtype=torch.long)
        print(edge_index_IBI)

        e_from_IOI = []
        e_end_IOI = []
        for i in range(len(data['IOI'])):
            for j in np.nonzero(data['IOI'][i])[0]:
                if i != j:
                    e_from_IOI.append(i)
                    e_end_IOI.append(j)
        edge_index_IOI = torch.tensor([e_from_IOI, e_end_IOI], dtype=torch.long)
        print(edge_index_IOI)

        e_from_ITI = []
        e_end_ITI = []
        for i in range(len(data['ITI'])):
            for j in np.nonzero(data['ITI'][i])[0]:
                if i != j:
                    e_from_ITI.append(i)
                    e_end_ITI.append(j)
        edge_index_ITI = torch.tensor([e_from_ITI, e_end_ITI], dtype=torch.long)
        print(edge_index_ITI)

        y = torch.tensor(data['label'], dtype=torch.float)
        print(y)

        #         x = []
        #         for i in data['train_idx'][0]:
        #             x.append(data['feature'][i])
        x = torch.tensor(data['feature'], dtype=torch.float)
        #         x = torch.tensor(np.array(x), dtype=torch.float)
        print(x)
        edge_index = torch.tensor([[0], [0]])
        edge_attr = torch.tensor([0])
        data = MyData(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, edge_index_IVI=edge_index_IVI,
                      edge_index_IBI=edge_index_IBI,edge_index_IOI=edge_index_IOI,edge_index_ITI=edge_index_ITI)

        torch.save(data, os.path.join(self.processed_dir, 'data.pt'))
    def len(self):
        return 1

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, "data.pt"))
        return data


dataset = MyDataset("./dataset")
data = dataset[0]
# print(len(data.node_sem))
# print(data.edge_index_MAM)
print(data)
print(len(dataset))


class MyGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 需要找一下这个2000维是哪出来的
        # 寄，这个2000是imdb数据集的自带特征维数，寄，这份代码把卷积层数量用到最大了
        # SAGE使用GraphSAGE的基于归纳聚类的方法进行卷积
        self.conv1 = SAGEConv(2000, 1024)
        self.linear1 = Linear(1024, 512)
        # GATConv使用的是基于注意力机制的图卷积网络
        self.conv2 = GATConv(512, 256)
        self.conv3 = GATConv(256, 128)
        # linear2 之后输出64维的向量表示，不能在加卷积层了，64维特征表示是论文测试出的最佳表示
        self.linear2 = Linear(128, 64)
        #         self.linear3 = Linear(128,64)
        # 这一层相当于分类了
        self.conv4 = SAGEConv(64, 4)
    def forward(self, data):
        x, edge_index_IVI, edge_index_IOI,edge_index_IBI,edge_index_ITI, = (
            data.x,
            data.edge_index_IVI,
            data.edge_index_IBI,
            data.edge_index_ITI,
            data.edge_index_IOI,
        )
        x = F.relu(self.conv1(x, edge_index_IVI))
        x = F.relu(self.linear1(x))

        # dropout需要放在需要dropout的网络层的前面
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, torch.cat((edge_index_IVI, edge_index_IBI,
            edge_index_IOI, edge_index_ITI), 1)))

        x = F.relu(self.conv3(x, torch.cat((edge_index_IVI, edge_index_IBI,
            edge_index_IOI, edge_index_ITI), 1)))
        x = F.relu(self.linear2(x))

        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv4(x, edge_index_IOI))

        return F.log_softmax(x, dim=1)
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list \
        ('cmap', ['#00FF00', '#F4A460', '#00CDCD', '#0000FF'], 256)

def visualize(out, color):
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

    # 设置窗口大小
    plt.figure(figsize=(7, 7))

    # 去掉坐标轴的刻度线
    plt.xticks([])
    plt.yticks([])

    # cm = colormap()
    plt.scatter(z[:, 0], z[:, 1], s=40, cmap="Set2", c=color, alpha=0.5)
    plt.show()


model = MyGCN()
print(model)
model.eval()
out = model(data)
visualize(out, color=data.y)


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["loss"], label="loss")

    ax1.set_ylim([0.3, 1.3])
    ax1.legend()
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")

    ax2.plot(history["acc"], label="accuracy")

    ax2.set_ylim([0.1, 1])
    ax2.legend()
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")

    fig.suptitle("Training History")
    plt.show()


history = defaultdict(list)
# history['acc'].append(accuracy)
# history['loss'].append(loss)
accuracy_ = []
loss_ = []

# 定义超参数
lr = 0.008
epochs = 300

# 加载数据集
data = dataset[0]
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 定义模型和优化器
model = MyGCN()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = torch.nn.CrossEntropyLoss()

# 训练模型
model.train()
# 设置学习率调整器在训练过程中自动调整学习率
scheduler = StepLR(optimizer, step_size=100, gamma=0.1, verbose=False)
for epoch in range(epochs):
    acc_train = 0
    correct_count_train = 0
    optimizer.zero_grad()
    out = model(data)
    #     print(len(out))
    #     print(len(data.y))
    correct_count_train = out.argmax(axis=1).eq(data.y.argmax(axis=1)).sum().item()
    loss = loss_function(out, data.y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_.append(loss.item())
    accuracy_.append(correct_count_train / data.x.shape[0])
    print(
        f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}，Acc: {correct_count_train/data.x.shape[0]}"
    )


history["acc"] = accuracy_
history["loss"] = loss_
plot_history(history)

model.eval()
correct_count_train = 0
out = model(data)
correct_count_train = out.argmax(axis=1).eq(data.y.argmax(axis=1)).sum().item()
print(f"Acc: {correct_count_train/data.x.shape[0]:.4f}")


model.eval()
out = model(data)
visualize(out, color=data.y)
