import numpy as np

# 创建一个长度为10的空向量
np.zeros(10)

# 找到任何一个数组的内存大小
x = np.zeros([10, 10])
s = x.itemsize * x.size

# 创建一个长度为10并且除了第五个值为1的空向量
x = np.zeros(10)
x[4] = 1

# 创建一个值域范围从10到49的向量
x = np.arange(10, 50)

# 反转一个向量(第一个元素变为最后一个)
x = np.arange(50)
x = x[::-1]

# 创建一个 3x3 并且值从0到8的矩阵
x = np.arange(9)
x = x.reshape([3, 3])

# 找到数组[1,2,0,0,4,0]中非0元素的位置索引
np.nonzero(np.array([1, 2, 0, 0, 4, 0]))

# 创建一个 3x3 的单位矩阵
np.eye(3)

# 创建一个 3x3x3的随机数组
np.random.random([3, 3, 3])

# 创建一个 10x10 的随机数组并找到它的最大值和最小值
x = np.random.random([10, 10])
np.max(x)
np.min(x)

# 创建一个长度为30的随机向量并找到它的平均值
x = np.random.random(30)
np.mean(x)

# 创建一二维数组，其中边界值为1，其余值为0
x = np.ones([10, 10])
x[1:-1, 1:-1] = 0

# 对于一个存在在数组，如何添加一个用0填充的边界
x = np.ones([5, 5])
np.pad(x, 1, 'constant', constant_values=0)

# 以下表达式运行的结果分别是什么
assert 0.3 != 3 * 0.1
assert np.nan * 0 == 0
assert (np.nan == np.nan) is False
assert (np.inf > np.nan) is False
assert np.nan - np.nan == np.nan

# 创建一个 5x5的矩阵，并设置值1,2,3,4落在其对角线下方位置
x = np.diag(1 + np.arange(4))
np.pad(x, ((1, 0), (0, 1)), 'constant')

# 创建一个8x8 的矩阵，并且设置成棋盘样式
x = np.zeros([8, 8])
x[::2, ::2] = 1
x[1::2, 1::2] = 1

# 考虑一个 (6,7,8) 形状的数组，其第100个元素的索引(x,y,z)是什么?
x = np.zeros([6, 7, 8])
np.unravel_index(100, x.shape)

# 用tile函数去创建一个 8x8的棋盘样式矩阵
np.tile([(0, 1), (1, 0)], (4, 4))

# 对一个5x5的随机矩阵做归一化
x = np.random.random([5, 5])
(x - x.min()) / (x.max() - x.min())

# 创建一个将颜色描述为(RGBA)四个无符号字节的自定义dtype
np.dtype([('r', np.ubyte, 1), ('g', np.ubyte, 1), ('b', np.ubyte, 1), ('a', np.ubyte, 1)])

# 一个5x3的矩阵与一个3x2的矩阵相乘，实矩阵乘积是什么？
x = np.random.random([5, 3])
y = np.random.random([3, 2])
x.dot(y)

# 给定一个一维数组，对其在3到8之间的所有元素取反
x = np.arange(20)
x[(x >= 3) & (x <= 8)] *= -1

# 29. 如何从零位对浮点数组做舍入 ?
Z = np.random.uniform(-10, +10, 10)
np.copysign(np.ceil(np.abs(Z)), Z)

# 如何找到两个数组中的共同元素?
X = np.random.randint(0, 10, 10)
Y = np.random.randint(0, 10, 10)
np.intersect1d(X, Y)

# emath
np.emath.sqrt(-1)

# 如何得到昨天，今天，明天的日期?
today = np.datetime64('today', 'D')
yesterday = today - np.timedelta64(1, 'D')
tomorrow = today + np.timedelta64(1, 'D')
print('{};{};{}'.format(yesterday, today, tomorrow))

# 如何得到所有与2016年7月对应的日期？
np.arange('2016-07', '2016-08', dtype='datetime64[D]')
np.arange('2016-07-01', '2016-07-02', dtype='datetime64[h]')

# 用五种不同的方法去提取一个随机数组的整数部分
A = np.random.uniform(0, 10, 10)
np.floor(A)
np.ceil(A)
A.astype(int)
np.trunc(A)

# 创建一个5x5的矩阵，其中每行的数值范围从0到4
np.tile(np.arange(0, 5), [5, 1])


# 通过生成器来创建数组
def generator(x):
    assert x > 0
    for i in range(0, x):
        yield i ** 2
np.fromiter(generator(10), dtype=float)

# 输出0~100数据的5个分位数
np.linspace(0, 100, 5)

# 创建一个长度为10的随机向量，并将其排序
np.sort(np.random.random(10))

# TODO 对于一个小数组，如何用比 np.sum更快的方式对其求和？
np.add.reduce(np.random.random(10))

# 对于两个随机数组A和B，检查它们是否相等
A = np.random.random(10)
B = np.random.random(10)
np.allclose(A, B)
np.array_equal(A, B)

# 创建一个只读数组(read-only)
X = np.random.random([10, 10])
X.flags.writeable = False

# 创建一个长度为10的向量，并将向量中最大值替换为1
Z = np.random.random(10)
Z[np.argmax(Z)] = 1

# TODO 创建一个结构化数组，并实现 x 和 y 坐标覆盖 [0,1]x[0,1] 区域
Z = np.zeros((5, 5), dtype=[('x', float), ('y', float)])
np.meshgrid()

# 给定两个数组X和Y，构造Cauchy矩阵C (Cij =1/(xi - yj))
X = np.arange(8)
Y = X + 0.5
C = 1 / np.subtract.outer(X, Y)

# 打印每个numpy标量类型的最小值和最大值？
ntypes = [np.int, np.int8, np.int32, np.int64]
for t in ntypes:
    print(np.iinfo(t).max)
    print(np.iinfo(t).min)

# 给定标量时，如何找到数组中最接近标量的值？
X = np.random.randint(0, 20, 10)
t = 6
X[np.abs(X - t).argmin()]

# TODO 创建一个表示位置(x,y)和颜色(r,g,b)的结构化数组
X = np.zeros(10, dtype=[('position', [('x', float), ('y', float)]),
                            ('color', [('r', int), ('g', int), ('b', int)])])

#  如何将32位的浮点数(float)转换为对应的整数(integer)
X = np.arange(1, 10, dtype=np.float32)
X.astype(np.int32)

# 对于numpy数组，enumerate的等价操作是什么？
X = np.arange(9).reshape(3, 3)
for position, x in np.ndenumerate(X):
    print(position, x)

# 56. 生成一个通用的二维Gaussian-like数组