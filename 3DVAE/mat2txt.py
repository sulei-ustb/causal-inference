# 这是一个开头
# 人员：Mr Su
# 开发时间：20/11/2020下午8:38
# 文件名：mat2txt.py
# 开发工具：PyCharm

def mat2txt():
    import scipy.io as sio
    import numpy as np
    data = sio.loadmat(r'C:\Users\Mr Su\Desktop\botong1.mat')

    # print('scipy读取单位矩阵的初步结果:\n%s\n' % data)

    result = data['warehouse']
    # np.savetxt('botong1','result')

    rsp_result = result.reshape((3000,2))
    train_data = rsp_result[0:2000,0:2]
    test_data = rsp_result[2000:3000,0:2]

    # print(rsp_result)
    # print(np.size(result))
    # np.save(file="botong.npy", arr=rsp_result)
    #
    # np.savetxt('botong1',rsp_result)
    # np.savetxt('train_data',train_data)
    # np.savetxt('test_data',test_data)
    return train_data, test_data

