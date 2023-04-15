import numpy as np
from sklearn import linear_model
import scipy.misc
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat


class KSVD(object):
    def __init__(self, n_components, max_iter, tol, n_nonzero_coefs):
        '''
        稀疏模型Y=DX，Y位样本矩阵，使用KSVD动态更新字典矩阵D和稀疏稀疏
        param n_components 字典所含原子数目
        param max_iter 最大迭代次数
        param tol 稀疏表示结果的容差
        param n_nonzero_coefs 稀疏度
        '''
        self.dictionary = None
        self.sparsecode = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs

    def _initialize(self, y):
        '''
        初始化字典矩阵
        '''
        '''
        用随机选取样本的方法来初始化字典矩阵,y=160*3060

        y_sample = y.T #y_sample=3060*160
        rand_arr = np.arange(y_sample.shape[0])#生成0-3059的数
        np.random.shuffle(rand_arr)#将其顺序打乱
        shape1 = y_sample[rand_arr[0:200]].T
        self.dictionary = shape1
        print(self.dictionary.shape)

        for i in range(200):
            self.dictionary[:, i] = self.dictionary[:, i] / np.linalg.norm(self.dictionary)

        '''
        '''
        用随机二阶单位范数矩阵来初始化字典矩阵
        '''

        shape = [y.shape[0], self.n_components]
        self.dictionary = np.random.random(shape)
        for i in range(shape[1]):
            self.dictionary[:, i] = self.dictionary[:, i] / np.linalg.norm(self.dictionary)

        '''u,s,v=np.linalg.svd(y)
        self.dictionary=u[:,:self.n_components]'''
        '''初始化矩阵，用u矩阵来初始化,初始样本y是n*M的矩阵，则u矩阵是n*n
        而我们要构建过完备字典n*K的要求是K>n

        '''

    def _update_dict(self, y, d, x):
        '''使用ksvd更新字典的过程
        '''
        for i in range(self.n_components):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue
            d[:, i] = 0  # 更新第i列
            r = (y - np.dot(d, x))[:, index]  # 计算误差矩阵
            # 利用svd，求解更新字典和稀疏系数
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0].T  # 使用左奇异矩阵的第一列更新字典
            x[i, index] = s[0] * v[0, :]
        return d, x

    def fit(self, y):
        '''
        迭代过程
        '''
        self._initialize(y)  # 初始化字典
        for i in range(self.max_iter):
            # i=1
            # while 1:
            # start2=time.clock()

            x = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
            # 使用omp算法来计算稀疏稀疏
            e = np.linalg.norm(y - np.dot(self.dictionary, x))
            # norm范数
            if e < self.tol:
                break
            self._update_dict(y, self.dictionary, x)
            # 使用svd进行字典更新
            # costtime=time.clock()-start2
            print('已经进行了%s次迭代，此时的字典误差为:%s' % (i, e))
            # i=i+1

        self.sparsecode = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
        return self.dictionary, self.sparsecode
    # 返回更新后的字典，和稀疏系数

    # 主程序


def error(A, B):
    sum = 0
    for i in range(200):
        for j in range(512):
            plus = ((A[i, j] - B[i, j]) ** 2) / (B[i, j] ** 2)
            sum = sum + plus

    '''
    C = A - B
    C = C.reshape(1, 102400)
    D = B.reshape(1, 102400)
    sum=0
    for i in range(102400):
        if D[0,i]==0:
            sum=sum+0
        else:
            sum=sum+((C[0,i]*C[0,i])/(D[0,i]*D[0,i]))
    e=sum*100
    return e
    '''
    return sum


if __name__ == '__main__':
    start = time.perf_counter()

    train_data = loadmat("D:\py\ksvd-sparse-dictionary-master\sample_inner.mat")
    train_data = train_data["sample_inner"]
    '''
    im_ascent=scipy.misc.ascent().astype(np.float)
    ksvd=KSVD(300)
    dictionary,sparsecode=ksvd.fit(im_ascent)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im_ascent)
    plt.subplot(1,2,2)
    plt.imshow(dictionary.dot(sparsecode))
    plt.show()
    '''

    ksvd = KSVD(300, 100, 1e-1, 10)
    # 输入的是构建的字典原子个数k，迭代次数100，误差和稀疏系数
    dictionary, sparsecode = ksvd.fit(train_data)
    # 计算此时和原来的误差
    re_data = np.dot(dictionary, sparsecode)

    e = error(re_data, train_data)

    print('此时的重构误差为%s' % (e), "%")
    train_data = train_data.T
    # 因为python中的reshape和matlab中的reshape重构顺序不一样，matlab是由列先，python是行先，所以转置一下。
    trian_data1 = train_data.reshape(1, 102400)
    re_data = re_data.T
    re_data1 = re_data.reshape(1, 102400)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(trian_data1[0, :])
    plt.subplot(2, 1, 2)
    plt.plot(re_data1[0, :])
    plt.show()
    # 打印一下字典中的原子，看下提取到冲击特征没有，但很遗憾，一直提取不到
    for i in range(100):
        plt.figure()
        plt.plot(dictionary[:, i])
        plt.title("第", '%s' % (i), "张图片")
        plt.ylim(-0.1, 0.1)
        plt.show()

    alltimecost = (time.clock() - start)
    print('总共花了%s秒' % (alltimecost))
    print(train_data.shape)

