# 这是一个开头
# 人员：Mr Su
# 开发时间：18/11/2020下午3:49
# 文件名：hw3_helper.py
# 开发工具：PyCharm
# from .utils import *
from mydocument.utils import *

from mydocument.mat2txt import *
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import  Axes3D


def plot_vae_training_plot(train_losses, test_losses, title, fname):

    # 改动前
    elbo_train, recon_train, kl_train = train_losses[:, 0], train_losses[:, 1], train_losses[:, 2]
    elbo_test, recon_test, kl_test = test_losses[:, 0], test_losses[:, 1], test_losses[:, 2]
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, elbo_train, label='-elbo_train')
    plt.plot(x_train, recon_train, label='recon_loss_train')
    plt.plot(x_train, kl_train, label='kl_loss_train')
    plt.plot(x_test, elbo_test, label='-elbo_test')
    plt.plot(x_test, recon_test, label='recon_loss_test')
    plt.plot(x_test, kl_test, label='kl_loss_test')

    plt.legend()
    # plt.title(title)
    plt.title('Dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    savefig(fname)

    # kl_train = train_losses
    # kl_test = test_losses
    # plt.figure()
    # n_epochs = len(test_losses) - 1
    # x_train = np.linspace(0, n_epochs, len(train_losses))
    # x_test = np.arange(n_epochs + 1)
    #
    # # plt.plot(x_train, elbo_train, label='-elbo_train')
    # # plt.plot(x_train, recon_train, label='recon_loss_train')
    # plt.plot(x_train, kl_train, label='train_loss')
    # # plt.plot(x_test, elbo_test, label='-elbo_test')
    # # plt.plot(x_test, recon_test, label='recon_loss_test')
    # plt.plot(x_test, kl_test, label='test_loss')
    #
    # plt.legend()
    # plt.title('Dataset')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # savefig(fname)




''' 一下4个def 分别为 样本生成函数'''
def sample_data_1_a(count):
    rand = np.random.RandomState(0)   #此处的0 为随机数生成的种子。note：只要种子相同，产生的随机数序列就相同
    # numpy的广播功能：https://www.runoob.com/numpy/numpy-broadcast.html，不同维度array进行操作方法
    # 此处使用方法 类似于（10000，2）的array 每行都加上一个 [[1.0,2.0]]

    # 输出为（count，2）的array
    # return [[1.0, 2.0]] + (rand.randn(count, 2) * [[5.0, 1.0]]).dot(
    #     [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])
    # 三维数据生成代码
    return [[1.0, 2.0, 3.0]] + (rand.randn(count, 3) * [[5.0, 1.0, 2.0]]).dot(
        [
            [np.sqrt(2) / 2, np.sqrt(2) / 2, np.sqrt(2) / 2],
            [-np.sqrt(2) / 2, np.sqrt(2) / 2, np.sqrt(2) / 2],
            [np.sqrt(2) / 2, -np.sqrt(2) / 2, np.sqrt(2) / 2]
         ])


def sample_data_2_a(count):
    rand = np.random.RandomState(0)
    #
    # return [[-1.0, 2.0]] + (rand.randn(count, 2) * [[1.0, 5.0]]).dot(
    #     [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])


    #改动后的三维数据生成代码
    return [[1.0, 2.0, 3.0]] + (rand.randn(count, 3) * [[5.0, 1.0, 2.0]]).dot(
        [
            [np.sqrt(2) / 2, np.sqrt(2) / 2, np.sqrt(2) / 2],
            [np.sqrt(2) / 2, -np.sqrt(2) / 2, np.sqrt(2) / 2],
            [np.sqrt(2) / 2, np.sqrt(2) / 2, -np.sqrt(2) / 2]
         ])


    # 以下为求的代码
    # return [[-0.2, 0.4]] + (rand.randn(count, 2) * [[1.0, 5.0]]).dot(
    #     [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])



def sample_data_1_b(count):
    rand = np.random.RandomState(0)
    return [[1.0, 2.0]] + rand.randn(count, 2) * [[5.0, 1.0]]


def sample_data_2_b(count):
    rand = np.random.RandomState(0)
    return [[-1.0, 2.0]] + rand.randn(count, 2) * [[1.0, 5.0]]

# 三维数据生成
# 生成时间 2020/11/28



def q1_sample_data(part, dset_id):
    assert dset_id in [1, 2]  #assert?  断言函数，理解成生成一个列表
    assert part in ['a', 'b'] # 判断 后面条件  是否为真
    if part == 'a':
        if dset_id == 1:
            dset_fn = sample_data_1_a  #  特殊的写法   f1 = f2；  如果对f2已有定义（def f(2) ）,可以对f1(count) 直接调用
        else:
            dset_fn = sample_data_2_a
    else:
        if dset_id == 1:
            dset_fn = sample_data_1_b
        else:
            dset_fn = sample_data_2_b
    # 改动前
    train_data, test_data = dset_fn(10000), dset_fn(2500)
    # train_data, test_data = dset_fn(10000), dset_fn(2500)
    return train_data.astype('float32'), test_data.astype('float32')


def visualize_q1_data(part, dset_id):
    # train_data, test_data = q1_sample_data(part, dset_id)   #返回 训练集10000个，测试集2500个
    # xs1, xs2 = train_data[:,0], test_data[:,0]
    # ys1,ys2 = train_data[:,1], test_data[:,1]
    # zs1,zs2 = train_data[:2], test_data[:,2]
    # # train_data, test_data = mat2txt()
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.set_title('Train Data')
    # ax1.scatter(train_data[:, 0], train_data[:, 1],train_data[:, 2])
    # ax2.set_title('Test Data')
    # ax2.scatter(test_data[:, 0], test_data[:, 1],test_data[:, 2])
    # print(f'Dataset {dset_id}')
    # plt.show()
    # 三维离散图绘制
    fig = plt.figure()
    ax = fig.gca(projection = "3d")
    plt.title('test')

    train_data, test_data = q1_sample_data(part, dset_id)   #返回 训练集10000个，测试集2500个
    xs1, xs2 = train_data[:,0], test_data[:,0]
    ys1,ys2 = train_data[:,1], test_data[:,1]
    zs1,zs2 = train_data[:,2], test_data[:,2]

    # ax.scatter(xs1,ys1,zs1,zdir = "z", c = "#00DDAA",markeer = "0",s = 40)
    ax.scatter(xs1,ys1,zs1,zdir = "z", c = "#00DDAA",marker = "o",s = 40)
    ax.scatter(xs2,ys2,zs2,zdir = "z", c = "#FF5511",marker = "^",s = 40)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")

    plt.show()

def q1_save_results(part, dset_id,fn):
    #train_data(10000,2);test_data(2500,2)
    train_data, test_data = q1_sample_data(part, dset_id)


    # np.savetxt('test_data', test_data)
    # np.savetxt('train_data', train_data)

    train_losses, test_losses, samples_noise, samples_nonoise = fn(train_data, test_data, part, dset_id)
    # 改动前
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')
    # 改动后
    # print(f'Final -ELBO: {test_losses[-1, 0]:.4f},')
    # 保存两组data
    np.savetxt('sample_noise',samples_noise)
    np.savetxt('sample_nonoise',samples_nonoise)

    # plot_vae_training_plot(train_losses, test_losses, f'Q1({part}) Dataset {dset_id} Train Plot',
    #                        f'results/q1_{part}_dset{dset_id}_train_plot.png')

    plot_vae_training_plot(train_losses, test_losses, f'Q1({part}) Dataset {dset_id} Train Plot',
                           f'results/q1_{part}_dset{dset_id}_train_plot.png')

    # save_scatter_2d(samples_noise, title='Samples with Decoder Noise',
    #                 fname=f'results/q1_{part}_dset{dset_id}_sample_with_noise.png')
    # save_scatter_2d(samples_nonoise, title='Samples without Decoder Noise',
    #                 fname=f'results/q1_{part}_dset{dset_id}_sample_without_noise.png')

#     3D绘图与保存
    save_scatter_3d(samples_noise, title='Samples with Decoder Noise',
                    fname=f'results/q1_{part}_dset{dset_id}_sample_with_noise.png')
    save_scatter_3d(samples_nonoise, title='Samples without Decoder Noise',
                    fname=f'results/q1_{part}_dset{dset_id}_sample_without_noise.png')


def visualize_colored_shapes():
    data_dir = get_data_dir(3)
    train_data, test_data = load_pickled_data(join(data_dir, 'shapes_colored.pkl'))
    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs]
    show_samples(images, title='Colored Shapes Samples')


def visualize_svhn():
    data_dir = get_data_dir(3)
    train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs]
    show_samples(images, title='SVHN Samples')


def visualize_cifar10():
    data_dir = get_data_dir(3)
    train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))
    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs]
    show_samples(images, title='CIFAR10 Samples')



def q2_save_results(part, dset_id, fn):
    assert part in ['a', 'b'] and dset_id in [1, 2]
    data_dir = get_data_dir(3)
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

    train_losses, test_losses, samples, reconstructions, interpolations = fn(train_data, test_data, dset_id)
    samples, reconstructions, interpolations = samples.astype('float32'), reconstructions.astype('float32'), interpolations.astype('float32')
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')
    plot_vae_training_plot(train_losses, test_losses, f'Q2({part}) Dataset {dset_id} Train Plot',
                           f'results/q2_{part}_dset{dset_id}_train_plot.png')
    show_samples(samples, title=f'Q2({part}) Dataset {dset_id} Samples',
                 fname=f'results/q2_{part}_dset{dset_id}_samples.png')
    show_samples(reconstructions, title=f'Q2({part}) Dataset {dset_id} Reconstructions',
                 fname=f'results/q2_{part}_dset{dset_id}_reconstructions.png')
    show_samples(interpolations, title=f'Q2({part}) Dataset {dset_id} Interpolations',
                 fname=f'results/q2_{part}_dset{dset_id}_interpolations.png')


def q3_save_results(dset_id, fn):
    assert dset_id in [1, 2]
    data_dir = get_data_dir(3)
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

    vqvae_train_losses, vqvae_test_losses, pixelcnn_train_losses, pixelcnn_test_losses, samples, reconstructions = fn(train_data, test_data, dset_id)
    samples, reconstructions = samples.astype('float32'), reconstructions.astype('float32')
    print(f'VQ-VAE Final Test Loss: {vqvae_test_losses[-1]:.4f}')
    print(f'PixelCNN Prior Final Test Loss: {pixelcnn_test_losses[-1]:.4f}')
    save_training_plot(vqvae_train_losses, vqvae_test_losses,f'Q3 Dataset {dset_id} VQ-VAE Train Plot',
                       f'results/q3_dset{dset_id}_vqvae_train_plot.png')
    save_training_plot(pixelcnn_train_losses, pixelcnn_test_losses,f'Q3 Dataset {dset_id} PixelCNN Prior Train Plot',
                       f'results/q3_dset{dset_id}_pixelcnn_train_plot.png')
    show_samples(samples, title=f'Q3 Dataset {dset_id} Samples',
                 fname=f'results/q3_dset{dset_id}_samples.png')
    show_samples(reconstructions, title=f'Q3 Dataset {dset_id} Reconstructions',
                 fname=f'results/q3_dset{dset_id}_reconstructions.png')


def q4_a_save_results(dset_id, fn):
    assert dset_id in [1, 2]
    data_dir = get_data_dir(3)
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

    vqvae_train_losses, vqvae_test_losses, pixelcnn_train_losses, pixelcnn_test_losses, samples, reconstructions = fn(train_data, test_data, dset_id)
    samples, reconstructions = samples.astype('float32'), reconstructions.astype('float32')
    print(f'VQ-VAE Final Test Loss: {vqvae_test_losses[-1]:.4f}')
    print(f'PixelCNN Prior Final Test Loss: {pixelcnn_test_losses[-1]:.4f}')
    save_training_plot(vqvae_train_losses, vqvae_test_losses,f'Q4(a) Dataset {dset_id} VQ-VAE Train Plot',
                       f'results/q4_a_dset{dset_id}_vqvae_train_plot.png')
    save_training_plot(pixelcnn_train_losses, pixelcnn_test_losses,f'Q4(a) Dataset {dset_id} PixelCNN Prior Train Plot',
                       f'results/q4_a_dset{dset_id}_pixelcnn_train_plot.png')
    show_samples(samples, title=f'Q4(a) Dataset {dset_id} Samples',
                 fname=f'results/q4_a_dset{dset_id}_samples.png')
    show_samples(reconstructions, title=f'Q4(a) Dataset {dset_id} Reconstructions',
                 fname=f'results/q4_a_dset{dset_id}_reconstructions.png')


def q4_b_save_results(fn):
    part = 'b'
    data_dir = get_data_dir(3)
    train_data, test_data = load_pickled_data(join(data_dir, 'mnist.pkl'))

    train_losses, test_losses, samples, reconstructions = fn(train_data, test_data)
    samples, reconstructions = samples.astype('float32') * 255, reconstructions.astype('float32') * 255
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')
    plot_vae_training_plot(train_losses, test_losses, f'Q4({part}) Train Plot',
                           f'results/q4_{part}_train_plot.png')
    show_samples(samples, title=f'Q4({part}) Samples',
                 fname=f'results/q4_{part}_samples.png')
    show_samples(reconstructions, title=f'Q4({part}) Reconstructions',
                 fname=f'results/q4_{part}_reconstructions.png')
