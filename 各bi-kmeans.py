#专门biKmeans
# ok的！https://www.cnblogs.com/MrLJC/p/4129700.html
def biKmeans(dataSet, k, distMeas=distEclud):
    #def distEclud(vecA, vecB):
    m =np.shape(dataSet)[0]
    clusterAssment =np.mat(np.zeros((m,2)))#记录簇分配的结果及误差
    centroid0 =np.mean(dataSet, axis=0).tolist()[0]#计算整个数据集的质心
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#计算初始聚类点与其他点的距离
        clusterAssment[j,1] = distMeas(np.mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = np.inf
        for i in range(len(centList)):#尝试划分每一簇
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
#             ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]
            print('ptsInCurrCluster.shape=',ptsInCurrCluster.shape)            
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2)#后面参数, distMeas)#对这个簇运行一个KMeans算法，k=2
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:##划分后更好的话
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #更新簇的分配结果change 1 to 3,4, or whatever
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print( 'the bestCentToSplit is: ',bestCentToSplit)
        print ('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return np.mat(centList), clusterAssment
	
	
#标准biKmeans
def biKmeans(dataSet, k):     #ok的
    """
    二分kmeans算法
    Args:
        dataSet: 数据集
        k: 聚类数
    Returns:
        centroids: 聚类中心
        clusterAssment: 点分配结果
    """
    m, n = np.shape(dataSet)
    # 起始时，只有一个簇，该簇的聚类中心为所有样本的平均位置
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    # 设置一个列表保存当前的聚类中心
    currentCentroids = [centroid0]
    # 点分配结果： 第一列指明样本所在的簇，第二列指明该样本到聚类中心的距离
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 初始化点分配结果，默认将所有样本先分配到初始簇
    for j in range(m):
        clusterAssment[j, 1] = distEclud(dataSet[j, :], np.mat(centroid0))**2
    # 直到簇的数目达标
    while len(currentCentroids) < k:
        # 当前最小的代价
        lowestError = np.inf
        # 对于每一个簇
        for j in range(len(currentCentroids)):
            # 获得该簇的样本
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0], :]
            print('ptsInCurrCluster.shape=',ptsInCurrCluster.shape)
            # 在该簇上进行2-means聚类
            # 注意，得到的centroids，其聚类编号含0，1
            centroids, clusterAss = kMeans(ptsInCurrCluster, 2)
            # 获得划分后的误差之和
            splitedError = np.sum(clusterAss[:, 1])
            '''
            # 获得其他簇的样本
            ptsNoInCluster = dataSet[np.nonzero(
                clusterAssment[:, 0].A != j)[0]]
            # 获得剩余数据集的误差
            nonSplitedError = np.sum(ptsNoInCluster[:, 1])
            '''
            # 获得剩余数据集的误差
            nonSplitedError = np.sum(clusterAssment[np.nonzero(
                clusterAssment[:, 0].A != j)[0]][:, 1])
            
            print("splitedError, and nonSplitedError: ",splitedError,nonSplitedError)
            # 比较，判断此次划分是否划算
            if (splitedError + nonSplitedError) < lowestError:
                # 记录当前的应当划分的簇
                needToSplit = j
                # 新获得的簇以及点分配结果
                newCentroids = centroids#.A
                newClusterAss = clusterAss.copy()
                # 如果划算，刷新总误差
                lowestError = splitedError + nonSplitedError


        # 更新簇的分配结果
        # 第1簇应当修正为最新一簇
        newClusterAss[np.nonzero(newClusterAss[:, 0].A == 1)[
            0], 0] = len(currentCentroids)
        # 第0簇应当修正为被划分的簇
        newClusterAss[np.nonzero(newClusterAss[:, 0].A == 0)[
            0], 0] = needToSplit
        print( 'the bestCentToSplit -needToSplit is: ', needToSplit)
        print ('the len of bestClustAss -newClusterAss is: ', len(newClusterAss))
        # 被划分的簇需要更新
        currentCentroids[needToSplit] = newCentroids[0, :].tolist()[0]#加了.tolist()[0]
        # 加入新的划分后的簇
        currentCentroids.append(newCentroids[1, :].tolist()[0])#加了.tolist()[0]
        # 刷新点分配结果
        clusterAssment[np.nonzero(
            clusterAssment[:, 0].A == needToSplit
        )[0], :] = newClusterAss
    return np.mat(currentCentroids), clusterAssment
	
	
#提出的改进二分聚类
	
def biKmeans_(dataSet, k):   #按：改进二分聚类，不需要每次新增一簇就对全部的旧簇反复做二分聚类。
    """
    二分kmeans算法
    Args:
        dataSet: 数据集
        k: 聚类数
    Returns:
        centroids: 聚类中心
        clusterAssment: 点分配结果
    """
    m, n = np.shape(dataSet)
    # 起始时，只有一个簇，该簇的聚类中心为所有样本的平均位置    
    #本簇的初始化，包括中心，所属，误差三项
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    # 设置一个列表保存当前的聚类中心
    currentCentroids = [centroid0]
    # 点分配结果： 第一列指明样本所在的簇，第二列指明该样本到聚类中心的距离 
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 初始化点分配结果，默认将所有样本先分配到初始簇
    for j in range(m):
        clusterAssment[j, 1] = distEclud(dataSet[j, :], np.mat(centroid0))**2
	#循环初始化新簇，补充的代码
    #新簇是为了二分支形成母簇而用
    
    ClusterErrorSum=[[]]#此表存放误差平方和SSE  不能是[]
    #子簇：
    #子簇（中心坐标，误差平方和SSE） SubClusters可分解的复合表 ，未使用
    #子簇的初始化，也包括中心，所属，误差三项   
    subClusterCentroids=[[]]#子簇中心坐标表
    subClusterAss=[[]]      #子簇所属表（所属子簇中心号，误差平方），可导出下列二项：
    subClusterErrorSum=[[]]  #各子簇的误差平方和
    splitedError=[[]]        #母簇分裂后的误差平方和，上述两子簇误差平方和相加
    ErrorDelta=[[]]  #初始误差平方和降低量

    #新生簇的初始化，也包括中心，所属，误差三项 ，为了和循环衔接  
    newClusterCentroids=currentCentroids 
    newClusterAssment=clusterAssment
    newClusterErrorSum=np.sum(newClusterAssment[:, 1]) #新簇的误差平方和
    Indexes_of_newClusterCentroids_In_currentCentroids=[0]#新生簇在当前簇集中的序号
#     ClusterErrorSum.append(newClusterErrorSum)#错误，改为下面的
    ClusterErrorSum[0].append(newClusterErrorSum)
    
#     辅助的历史记录表   在母簇分裂时更新
#     discenter=[] #可能直接用不上，是复合表
    #初值可取[newClusterCentroids,newClusterErrorSum]   
    #存储有：中心点，含有的误差平方和，母子分支的误差平方和的降低量，分裂后的残差平方和
    #disCenterIDs表，disCentroids表，disErrorSum表，disErrorDelta表，dissplitedError表
    disCenterIDs=[]
    disCentroids=[]
    disErrorSum=[]
    disErrorDelta=[]
    dissplitedError=[]
    
    needToSplit = 0  #提出来
 
    # 直到簇的数目达标
    while len(currentCentroids) < k:
    # 直到簇的数目达标
#     if len(currentCentroids) < k:    #按：要<而不是≤，循环结束时就已经生成全部k个中心点，k-1+1=k个簇
        # 当前最小的代价
        highestErrorDelta = 0
        # 对于每一个簇    #按：改为，对每一个新簇，存储二分聚类的两个子簇，还有误差平方和及其二分聚类的降低量。    
    
        for j in Indexes_of_newClusterCentroids_In_currentCentroids:
            # 获得该簇的样本           
            ptsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0], :]
            # 在该簇上进行2-means聚类
            newClusterPre=newCluster(ptsInCluster,2)      
            #返回subClusterCentroids,splitedError,subClusterErrorSum,subclusterAss
            print('type(newClusterPre[3])=',type(newClusterPre[3]))
            #按：测试结果： type(newClusterPre[3])= <class 'numpy.matrixlib.defmatrix.matrix'>
            subClusterCentroids[j].extend(newClusterPre[0])#更新j的[] 
            #按：要注意append和extend的区别！extend针对列表合并，append针对元素的追加
            print('append的subClusterCentroids=',subClusterCentroids[j])
            #append的subClusterCentroids= [[matrix([[-0.2897198 , -2.83942545]])]]
            splitedError[j].append(newClusterPre[1])
            subClusterErrorSum[j].append(newClusterPre[2])
#             subClusterAss[j].append(newClusterPre[3]) #按：[j]不可省掉，否则反复乱下去。重点细节
            subClusterAss[j].append(newClusterPre[3]) #  不宜用extend，否则得到一行行materix碎片
            print('ClusterErrorSum[',j,'][0]=',ClusterErrorSum[j][0])
            print('splitedError[',j,'][0]=',splitedError[j][0]) 
            ErrorDelta[j].append((ClusterErrorSum[j][0]-splitedError[j][0]))

        # 比较，判断此次划分是否划算
        for j in range(len(currentCentroids)):	
#             needToSplit = j#不能放在此处，否则大块的簇分割不掉！
            if ErrorDelta[j][0] >highestErrorDelta:# & len(currentCentroids)>1:
                # 如果还有更大降低量，刷新误差平方和的最大降低量
                highestErrorDelta = ErrorDelta[j][0]
                # 记录当前的应当划分的簇
                needToSplit = j
                # 新获得的簇以及点分配结果  #按：选拔要分裂的那个母簇
#         newCentroids =currentCentroids[needToSplit] #按：两组子簇中心按子簇号0、1排列   错误，这是单簇了
        newCentroids =subClusterCentroids[needToSplit] #按：两组子簇中心按子簇号0、1排列
        print('newCentroids=',newCentroids)
        print('type of newCentroids=',type(newCentroids))
        newClusterAssment =subClusterAss[needToSplit][0] 
#         print('newClusterAssment=',newClusterAssment)
        print('工作点 type of newClusterAssment=',type(newClusterAssment))
        #按：newClusterAssment =subClusterAss[needToSplit]的
        #测试结果为 type of newCentroids= <class 'list'>，后面的矩阵用法要加上[0]
        print('len of newClusterAssment=',len(newClusterAssment))
        #重点：找到错误根源所属关系矩阵或表，不能直接用needToSplit来给，它不是序号或行号，而是元素的首位数值
        print('clusterAssment[np.where(clusterAssment[:,0]==needToSplit)[0],:].shape=',
              clusterAssment[np.where(clusterAssment[:,0]==needToSplit)[0],:].shape)        
        print('currentCentroids=',currentCentroids)
        print('len(currentCentroids)=',len(currentCentroids),'选第',needToSplit,'号母簇')
        print('下面的0、1子簇，中心为：',newCentroids)
        '''
        '''
        # 第1簇应当修正为最新一簇
        newClusterAssment[np.nonzero(newClusterAssment[:, 0].A == 1)[
            0], 0] =len(currentCentroids) #TypeError: list indices must be integers or slices, not tuple
        # 第0簇应当修正为被划分的簇
        newClusterAssment[np.nonzero(newClusterAssment[:, 0].A == 0)[  #按：np.nonzero()，()内非零的逻辑判断，取逻辑真。
            0], 0]=needToSplit  

        #以下两行废弃，改为第三行
        #newClusterErrorSum=[]
        #newClusterErrorSum.extend(subClusterErrorSum[needToSplit])#注意：这里不是append,不针对元素，而是列表
        newClusterErrorSum=subClusterErrorSum[needToSplit][0]
        #仿效 newClusterAssment =subClusterAss[needToSplit][0]，不用下面的
#         newClusterErrorSum.append(subClusterErrorSum[needToSplit])#注意：这里不是append,不针对元素，而是列表
        print('newClusterErrorSum=',newClusterErrorSum,'for needToSplit=',needToSplit)
        #测得newClusterErrorSum= [[[466.63278133614426], [326.28407520118242]]] for needToSplit= 0

        
        #本簇更新：分裂的母簇更新为新簇：
        #保存历史记录（中心号，中心点坐标，误差项）
        #依次更新两子簇的：误差项，从属关系，中心点
        
        # 1/4        分裂的母簇更新之前保存历史记录      
        disCenterIDs.append(needToSplit)
        disCentroids.append(currentCentroids[needToSplit])
        disErrorSum.append(ClusterErrorSum[needToSplit])
        disErrorDelta.append(ErrorDelta[needToSplit])#按：其更新在新簇二分割聚类生成母簇时进行
        dissplitedError.append(splitedError[needToSplit])#按：其更新在新簇二分割聚类生成母簇时进行
        
        #2/4 误差项更新   注意区分使用和生成（更新）
        #检查，已做掉的：
        #ErrorDelta（确定分裂母簇时使用）
        #splitedError、subClusterErrorSum 各子簇的误差平方和  新簇二分支形成母簇时使用，生成则在母簇分裂时
        print('newClusterErrorSum[0]=',newClusterErrorSum[0])#测得 newClusterErrorSum[0]= [32.601242864951153]
        ClusterErrorSum[needToSplit]=newClusterErrorSum[0]
        ClusterErrorSum.append(newClusterErrorSum[1])        
        #以下的要加上
        splitedError[needToSplit]=[]
        splitedError.append([])
        subClusterErrorSum[needToSplit]=[]
        subClusterErrorSum.append([]) 
        ErrorDelta[needToSplit]=[]
        ErrorDelta.append([]) 
        
        
        
        #3/4 从属关系更新    包括本簇的和分裂的子簇的
        #本簇的从属关系更新
        # 刷新点分配结果    #按：所属簇号  即从属关系更新
        clusterAssment[np.nonzero(
            clusterAssment[:, 0].A == needToSplit
        )[0], :] = newClusterAssment  
        print('np.unique(newClusterAssment[:, 0].A)=',np.unique(newClusterAssment[:, 0].A))
        print('np.unique(clusterAssment[:, 0].A)=',np.unique(clusterAssment[:, 0].A ))
        #按：原簇簇号为needToSplit的样本被新的二分簇族（注意：是族）
        #                                             newClusterAss取代
        #簇号不变  
        #按：discenter表示被分裂的一代代母簇
        #存储有：中心点，含有的误差平方和，母子分支的误差平方和的降低量
        #子簇的从属关系更新
        subClusterAss[needToSplit]=[]  #不能是[[]]，否则出错
        subClusterAss.extend([[]])
        
        #4/4 中心点更新,包括本簇中心点更新currentCentroids和子簇中心点更新subClusterCentroids（重点：不能忽视）
        #本簇中心点更新currentCentroids
#         print('currentCentroids[needToSplit]=',currentCentroids[needToSplit])
         # 被划分的簇需要更新
        currentCentroids[needToSplit] = newCentroids[0]#.tolist()[0]#加了.tolist()[0]
        # 加入新的划分后的簇
        print('newCentroids[1]=',newCentroids[1]) #按：newCentroids[1]是单坐标点的列表[[单坐标点]]
        #newCentroids[1]= [[-3.38237045 -2.9473363 ]]  #返回一单行的矩阵[[]]  #重点关注，矩阵而非列表
#         currentCentroids.extend(newCentroids[1])#.tolist()[0])#加了.tolist()[0] #按：注意append和extend差别
        currentCentroids.append(newCentroids[1])#.tolist()[0]
        print('currentCentroids[-1]',currentCentroids[-1])
        print('currentCentroids[-1][0]',currentCentroids[-1][0])
        #子簇中心点更新subClusterCentroids
        subClusterCentroids[needToSplit]=[]
#         subClusterCentroids.extend([])  #按：考察功底的时刻
#         subClusterCentroids.extend([])    #按：等于什么也没干
        subClusterCentroids.extend([[]])  #按：增加了空位，即插入了[]，该序号i是可以用[i]append()来补充元素的
#         subClusterCentroids.append([[]])  #按：增加了元素[[]]，插入的是空元素的单元素表
        print('subClusterCentroids=',subClusterCentroids)
      
  
        #按：一对分裂的子簇变成一对新生簇，内部二分割变成待选分裂的母簇
        Indexes_of_newClusterCentroids_In_currentCentroids=[needToSplit,len(currentCentroids)-1]
        print('Indexes_of_newClusterCentroids_In_currentCentroids=',Indexes_of_newClusterCentroids_In_currentCentroids)    
        print('currentCentroids=',currentCentroids)
        print('#####################################################################################')

    #不堆叠，k类不会分开展示。位置放在最后return前更新更适宜  注意与return并头
    currentCentroids=np.vstack(currentCentroids)
    return np.mat(currentCentroids), clusterAssment
	
	
	
	