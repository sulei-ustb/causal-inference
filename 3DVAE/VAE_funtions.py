# 这是一个开头
# 人员：Mr Su
# 开发时间：18/11/2020下午8:43
# 文件名：VAE_funtions.py
# 开发工具：PyCharm


from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from mydocument.hw3_helper import *

def train(model, train_loader, optimizer, epoch, quiet, grad_clip=None):
    model.train()

    if not quiet:
        pbar = tqdm(total=len(train_loader.dataset))
    losses = OrderedDict()

    for x in train_loader:
        x = x.cuda()
        out = model.loss(x)
        optimizer.zero_grad()
        out['loss'].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        desc = f'Epoch {epoch}'
        for k, v in out.items():
            if k not in losses:
                losses[k] = []
            losses[k].append(v.item())
            avg_loss = np.mean(losses[k][-50:])
            desc += f', {k} {avg_loss:.4f}'

        if not quiet:
            pbar.set_description(desc)
            pbar.update(x.shape[0])
    if not quiet:
        pbar.close()
    return losses


def eval_loss(model, data_loader, quiet):
    model.eval()
    total_losses = OrderedDict()
    with torch.no_grad():
        for x in data_loader:
            x = x.cuda()
            out = model.loss(x)
            for k, v in out.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        desc = 'Test '
        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
            desc += f', {k} {total_losses[k]:.4f}'
        if not quiet:
            print(desc)
    return total_losses



# model = FullyConnectedVAE(2, 2, [128, 128], [128, 128]).cuda()
# train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
# test_loader = data.DataLoader(test_data, batch_size=128, shuffle=True)
#train_args = dict(epochs=10, lr=1e-3) ;字典格式进行查询；epochs:训练次数，lr ：learning rot

def train_epochs(model, train_loader, test_loader, train_args, quiet=False):
    epochs, lr = train_args['epochs'], train_args['lr'] # 次数与学习率的赋值
    grad_clip = train_args.get('grad_clip', None)#查找字典中是否有'grad_clip'，如果没有，则返回默认值 None
                                                #'grad_clip' 梯度剪切? 理解为梯度的一个阈值，通常人为设置，当梯度过大时，往往产生不好的后果？？
                                                #见：https://zhuanlan.zhihu.com/p/112904260
    # print(grad_clip)
    # torch.optim 优化算法之 optim.Adam();https://blog.csdn.net/kgzhang/article/details/77479737;
    #https://blog.csdn.net/weixin_38145317/article/details/104775536?utm_medium=distribute.pc_relevant.none-task-blog-title-7&spm=1001.2101.3001.4242
    # optimizer 为所建的优化对象，学习率为lr（0.001），用来保存当前状态并能够根据计算得到的梯度更新参数
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = OrderedDict(), OrderedDict()
    # OrderedDict 自动排序字典（根据字典对象中的元素），https://www.cnblogs.com/notzy/p/9312049.html

    for epoch in range(epochs):
        print("训练的次数是 ",epoch)
        model.train()
        train_loss = train(model, train_loader, optimizer, epoch, quiet, grad_clip)

        test_loss = eval_loss(model, test_loader, quiet)

        for k in train_loss.keys():
            if k not in train_losses:
                train_losses[k] = []
                test_losses[k] = []
            train_losses[k].extend(train_loss[k])
            # test_losses[k].append(test_loss[k])
            #改动后，10*test_loss
            test_losses[k].append(test_loss[k])

    return train_losses, test_losses


class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, hiddens=[]):
        super().__init__()

        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        if isinstance(output_shape, int):
            output_shape = (output_shape,)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hiddens = hiddens

        model = []
        prev_h = np.prod(input_shape)
        for h in hiddens + [np.prod(output_shape)]:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.ReLU())
            prev_h = h
        model.pop()
        self.net = nn.Sequential(*model)

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, -1)
        return self.net(x).view(b, *self.output_shape)

# model = FullyConnectedVAE(2, 2, [128, 128], [128, 128]).cuda()
'''

input_dim = 2
latent_dim = 2

encoder_hidden_sizes = [128, 128]

decoder_hidden_sizes = [128, 128]

??  输入层，隐含层，输出层。输入的维数为2，输出的维数为4，隐含层为128×128的，所以，
故，输入层为2→128，隐含层为128→128，输出层为128→4

'''
class FullyConnectedVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, enc_hidden_sizes=[],
                 dec_hidden_sizes=[]):
        super().__init__()

        # model = FullyConnectedVAE(3, 2, [128, 128], [128, 128]).cuda()
        self.latent_dim = latent_dim
        self.encoder = MLP(input_dim, 2 * latent_dim, enc_hidden_sizes)

        # self.encoder = MLP(input_dim = 2 , out_features = 4,encoder_hidden_sizes = [128, 128])
        self.decoder = MLP(latent_dim, 2 * input_dim, dec_hidden_sizes)
        # self.decoder = MLP(input_dim = 2 , out_features = 4,decoder_hidden_sizes = [128,128 ])

    def loss(self, x):
        a = self.encoder(x)
        # print(a.shape)
        mu_z, log_std_z = a.chunk(2, dim=1)# dim 表示对第二个维度向量进行分块
        # print(log_std_z.shape)
        z = torch.randn_like(mu_z) * log_std_z.exp() + mu_z
        # torch.randn_like(mu_z)生成的e1，e2，e3，等
        # print(z.shape)
        mu_x, log_std_x = self.decoder(z).chunk(2, dim=1)

        # Compute reconstruction loss - Note that it may be easier for you
        # to use torch.distributions.normal to compute the log_prob
        # 计算 recon_loss
        recon_loss = 0.5 * np.log(2 * np.pi) + log_std_x + \
                     (x - mu_x) ** 2 * torch.exp(-2 * log_std_x) * 0.5
        recon_loss = recon_loss.sum(1).mean()

        # Compute KL
        kl_loss = -log_std_z - 0.5 + (torch.exp(2 * log_std_z) + mu_z ** 2) * 0.5
        kl_loss = kl_loss.sum(1).mean()
        # 以下为改动前
        return OrderedDict(loss=recon_loss + kl_loss, recon_loss=recon_loss,
                           kl_loss=kl_loss)
        # 以下为改动后
        return OrderedDict( loss=kl_loss)

    def sample(self, n, noise=True):
        # n = 1000,生成1000个采样数据
        with torch.no_grad():         # 释放显存，不记录梯度信息
            z = torch.randn(n, self.latent_dim).cuda()  #
            # print(z)
            tem_num = self.decoder(z).chunk(2, dim=1)
            mu, log_std = tem_num
            if noise:
                z = torch.randn_like(mu) * log_std.exp() + mu
                # ci = exp(log_std)*ei + mu;  ei 服从正态分布

            # torch.randn_like(input) 相当于 torch.randn(input.size(),dtype = input.dtype,laout = input.layout,
            #                                            device = input.device)
            # 理解为生成一个与mu同尺寸，同类型且服从标准正态分布的tensor
            else:
                z = mu
        return z.cpu().numpy()

def q1(train_data, test_data, part, dset_id):
    """
    train_data: An (n_train, 2) numpy array of floats,（10000，2）
    test_data: An (n_test, 2) numpy array of floats, (2500,2)

    (You probably won't need to use the two inputs below, but they are there
     if you want to use them)
    part: An identifying string ('a' or 'b') of which part is being run. Most likely
          used to set different hyperparameters for different datasets
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a numpy array of size (1000, 2) of 1000 samples WITH decoder noise, i.e. sample z ~ p(z), x ~ p(x|z)
    - a numpy array of size (1000, 2) of 1000 samples WITHOUT decoder noise, i.e. sample z ~ p(z), x = mu(z)
    """

    """ YOUR CODE HERE """
    #
    # class FullyConnectedVAE(nn.Module):
    #     def __init__(self, input_dim, latent_dim, enc_hidden_sizes=[],
    #                  dec_hidden_sizes=[]):
    #         super().__init__()
    #         self.latent_dim = latent_dim
    #         self.encoder = MLP(input_dim, 2 * latent_dim, enc_hidden_sizes)
    #模型重新定义前
    # model = FullyConnectedVAE(2, 2, [128, 128], [128, 128]).cuda()
    # 模型重新定义后，输入为三维的tensor
    model = FullyConnectedVAE(3, 2, [128, 128], [128, 128]).cuda()

    print(model)
    #对数据进行加载，batch_size 表示每批训练的样本数目，shuffle 表示是否随机取样本
    #我的理解，train_data 即为随机从数据集中选择/装载出128个数据点作为训练样本（注意随机选取）
    #print(train_loder)的可能是一个指针，指向选择的起点？？？
    train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)

    test_loader = data.DataLoader(test_data, batch_size=128)
    # 改动学习率 前（）
    train_losses, test_losses = train_epochs(model, train_loader, test_loader,
                                             dict(epochs=10, lr=1e-3), quiet=True)
    # #改动后
    # train_losses, test_losses = train_epochs(model, train_loader, test_loader,
    #                                          dict(epochs=10, lr=1e-3), quiet=True)
    #改动前
    train_losses = np.stack((train_losses['loss'], train_losses['recon_loss'], train_losses['kl_loss']), axis=1)
    test_losses = np.stack((test_losses['loss'], test_losses['recon_loss'], test_losses['kl_loss']), axis=1)




    samples_noise = model.sample(1000, noise=True)
    samples_nonoise = model.sample(1000, noise=False)

    return train_losses, test_losses, samples_noise, samples_nonoise
'''
输出结果为
train_losses = {'loss':[list:790],
                'recon_loss':[list:790]
                'kl_loss':[list:790]
               }
test_losses = {'loss':[list:20],
                'recon_loss':[list:20]
                'kl_loss':[list:20]
               }
'''