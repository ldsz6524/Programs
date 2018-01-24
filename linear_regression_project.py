
# coding: utf-8

# In[ ]:


# 任意选一个你喜欢的整数，这能帮你得到稳定的结果
seed = 4


# In[22]:


#TODO 创建一个 4*4 单位矩阵
I = [[1,0,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]]


# In[25]:


# TODO 返回矩阵的行数和列数
def shape(M):
    return (len(M),len(M[0]))


# In[1]:


# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    for i in range(len(M)):
        for j in range(len(M[i])):
            M[i][j] = round(M[i][j],decPts)
    pass


# In[35]:


# TODO 计算矩阵的转置
def transpose(M):
    return [list(col) for col in zip(*M)]


# In[ ]:


# TODO 计算矩阵乘法 AB，如果无法相乘则raise ValueError
def matxMultiply(A, B):
    
    if len(A[0]) != len(B) :
        raise ValueError('fail')
    
    result = [[0] * len(B[0]) for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
            result[i][j] += A[i][k] * B[k][j]
                     
    return result 


# In[4]:


def augmentMatrix(A, b):
    return [x + y for x,y in zip(A,b)]


# In[2]:


# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    n = M[r1]
    M[r1] = M[r2]
    M[r2] = n
    pass


# In[5]:


# TODO r1 <--- r1 * scale
# scale为0是非法输入，要求 raise ValueError
# 直接修改参数矩阵，无返回值
def scaleRow(M, r1, scale):
    if scale != 0:
        M[r1] = [x * scale for x in M[r1]]
    else:
        raise ValueError('fail')
    pass


# In[6]:


# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    M[r1] = [x + y * scale for x,y in zip(M[r1],M[r2])]
    pass


# In[9]:


# TODO 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""

def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    if len(A) != len(b) : 
        return None 
    Ab = augmentMatrix(A, b)

    for c in range(len(A)):
        col = transpose(Ab)[c][c:] 
        max_val = max(col, key=abs)
        if abs(max_val) < epsilon: 
            return None 
        max_idx = col.index(max_val) + c 

        swapRows(Ab, c, max_idx) 
        scaleRow(Ab, c, 1.0 / Ab[c][c])

        for i in range(len(Ab)):
            if i != c and Ab[i][c] != 0:
                addScaledRow(Ab,i,c,-Ab[i][c])

    result = [[row[-1]] for row in Ab]
    matxRound(result, decPts)

    return result


# In[10]:


#TODO 请选择最适合的直线 y = mx + b
m1 = -5.1
b1 = 11.9


# In[ ]:


# TODO 实现以下函数并输出所选直线的MSE
def calculateMSE(X,Y,m,b):
    if len(X) == len(Y) and len(X) != 0:
        n = len(X)
        sum_list = [(Y[i] - m * X [i] - b) ** 2 for i in range(n)]
        return sum(sum_list) / n
    else:
        raise ValueError
print(calculateMSE(X,Y,m1,b1))


# In[ ]:


# TODO 实现线性回归
'''
参数：X, Y 存储着一一对应的横坐标与纵坐标的两个一维数组
返回：m，b 浮点数
'''
def linearRegression(X,Y):
    X = [[x, 1] for x in X]
    Y = [[y] for y in Y]
    XT = transpose(X)
    A = matxMultiply(XT, X)
    b = matxMultiply(XT, Y)
    result_list = gj_Solve(A, b)
    return result_list[0][0], result_list[1][0]

m2,b2 = linearRegression(X,Y)
assert isinstance(m2,float),"m is not a float"
assert isinstance(b2,float),"b is not a float"
print(m2,b2)

