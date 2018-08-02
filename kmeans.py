# %load kmeans.py
# kmeans.py
import numpy as np

def loadDataSet(filename):
    """
    
    读取数据集

    Args:
        filename: 文件名
    Returns:
        dataMat: 数据样本矩阵
    """
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 通过map函数批量转换
        #fitLine = map(float, curLine)#py3中改为
        fitLine =list(map(float, curLine))
        dataMat.append(fitLine)
    return dataMat

def distEclud(vecA, vecB):
    """
    计算两向量的欧氏距离

    Args:
        vecA: 向量A
        vecB: 向量B
    Returns:
        欧式距离
    """
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataSet, k):
    """
    随机生成k个聚类中心

    Args:
        dataSet: 数据集
        k: 簇数目
    Returns:
        centroids: 聚类中心矩阵
    """
    _, n = dataSet.shape
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        # 随机聚类中心落在数据集的边界之内
        minJ = np.min(dataSet[:, j])
        maxJ = np.max(dataSet[:, j])
#         print('minJ =',minJ )
        rangeJ = float(maxJ - minJ)
        
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, maxIter = 5):
    """
    K-Means

    Args:
        dataSet: 数据集
        k: 聚类数
    Returns:
        centroids: 聚类中心
        clusterAssment: 点分配结果
    """
    # 随机初始化聚类中心
    centroids = randCent(dataSet, k)
    m, n = np.shape(dataSet)
    # 点分配结果： 第一列指明样本所在的簇，第二列指明该样本到聚类中心的距离
    clusterAssment = np.mat(np.zeros((m, 2)))#按：开始，各子簇簇号都是0号
    # 标识聚类中心是否仍在改变
    clusterChanged = True
    # 直至聚类中心不再变化
    #按：补充，发现空簇
    kong=k*[0]#各簇 计数，初始值0
    iterCount = 0
    while clusterChanged and iterCount < maxIter:
        iterCount += 1
        clusterChanged = False
        # 分配样本到簇
        for i in range(m):
            # 计算第i个样本到各个聚类中心的距离
            minIndex = 0
            minDist = np.inf
            for j in range(k):
                dist = distEclud(dataSet[i, :],  centroids[j, :])
                if(dist < minDist):
                    minIndex = j
                    minDist = dist
            kong[minIndex]=kong[minIndex] +1       #各簇计数 
            # 判断cluster是否改变
            if(clusterAssment[i, 0] != minIndex):
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        
        #判断是否有空簇  补充代码
#         for j in range(k):
#             if kong[j]==0:
#                 print('出现第',j,'号空簇')
        # 刷新聚类中心: 移动聚类中心到所在簇的均值位置
        #空簇号kongids
        kongids=[]
        for cent in range(k):  #按：子簇在母簇内部编号cent
            #过滤空簇   补充代码
#             if kong[cent]==0:
#                 print('出现第',j,'号空簇，现在跳过')
#                 kongids.append(cent)#加入空簇号列表
#                 continue
            # 通过数组过滤获得簇中的点
            ptsInCluster = dataSet[np.nonzero(
                clusterAssment[:, 0].A == cent)[0]]  
            #按：clusterAssment[:, 0].A变单列矩阵为数组A
            if ptsInCluster.shape[0] > 0:
                # 计算均值并移动     
                #按：kMeans的由来 适用于数值型变量，本质是取序。
                #因此，为避免极端值对均值的影响，kMeans不是不能用，而是要经过每维序号化预处理。
                #序号化的中值设定为0，小端为负，大端为正。
                centroids[cent, :] = np.mean(ptsInCluster, axis=0) #注： centroids是矩阵,本行可见
    return centroids, clusterAssment#,kong,kongids #分别进行频率计数和空簇统计  
def newCluster(ptsInCluster,k=2):#母簇ptsInCluster生成k=2分割的子簇,母簇号不用携带
    #预备抽象的模块newCluster(ptsInCluster,k),
    #返回newCluster=[newClusterCentroids,subClusterCentroids,splitedError,ErrorDelta,subclusterAss]
    centroids, clusterAss = kMeans(ptsInCluster, k)
    #按：此步为优化考虑点，不需要每次新增一簇就对全部的旧簇反复做二分聚类，仅仅针对新增的簇做二分聚类，并保存。
    # 获得划分后的误差之和
    
    #不再承担老簇的统计，只需要统计本簇的新残差平方和
#     m=len(ptsInCluster)
#     centroid0 = np.mean(ptsInCluster, axis=0).tolist()[0]
#     clusterAssment = np.mat(np.zeros((m, 2)))
#     for j in range(m):
#         clusterAssment[j, 1] = distEclud(ptsInCluster[j, :], np.mat(centroid0))**2             
#     oldClusterAssment=clusterAssment
#     oldClusterErrorSum=np.sum(oldClusterAssment[:, 1])  
            
    splitedError= np.sum(clusterAss[:, 1])#新残差平方和
#     ErrorDelta=oldClusterErrorSum-splitedError#不承担统计此功能
#[[np.sum([a[i][1] for i in  range(len(a)) if a[i][0]==j])] for j in [0,1]] #可行的参考
    subClusterErrorSum=[[np.sum([clusterAss[i,1] for i in  range(len(clusterAss)) 
                                 if clusterAss[i,0]==j])] for j in [0,1]]   
    subClusterCentroids=centroids
    subClusterAss=clusterAss

    #以下两个等效动作是分裂动作，从子簇身份生成新簇，这是在选拔过程中完成的
    #本过程是要确定母子分支，当好母簇。
    #动作1
#     newCluster0=[subClusterCentroids.tolist()[0],[],splitedError,ErrorDelta,subclusterAss]
#     newCluster1=[subClusterCentroids.tolist()[1],[],splitedError,ErrorDelta,subclusterAss]
#     newCluster=[newCluster0,newCluster1]
    #动作2
#     newCluster=[]
#     for i in [0,1]:
#         print(subClusterCentroids[i],'/n',i)        
#         newCluster.append([subClusterCentroids[i],[],splitedError,ErrorDelta,subclusterAss[i]])
#         #extend是对列表；append是对元素
    #正确动作：返回母子分支，当好母簇
#     newCluster=[[],subClusterCentroids,splitedError,ErrorDelta,subclusterAss]
    #ErrorDelta不在放入了，改为subClusterErrorSum
    newCluster=[subClusterCentroids,splitedError,subClusterErrorSum,subClusterAss]
    print('type(subClusterCentroids)=',type(subClusterCentroids))
    print('本身type(subClusterAss)=',type(subClusterAss))#矩阵
                        #但在优化的。。。，下游，缺失#type of newClusterAssment= <class 'list'>
    
    return newCluster  