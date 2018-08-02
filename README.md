 
# improved-bi-kmeans
改进的二分聚类 选用的优化目标：母簇分裂的误差平方和降低量


## 主题概念（略）  

## 主要思想  
### 单一过程来看原理： 关键词是 母簇 子簇 新簇 回到母簇，区分本簇（现有的簇）    
现有的母簇都有母子分支，从候选集（本簇也是母簇）中选拔误差平方和降低量最大的母簇做分裂，聚类个数增加1。  
母簇分裂得到两子簇，取代原来的母簇，正式成为新簇。  
聚类个数达到要求，则可以提交分裂结果。新簇内部还没有二分聚类，如果聚类个数不够，新簇如循环初始一样，继续二分聚类，成为新生的母簇。  
每次分裂的两个新簇各自二分聚类，生成两簇母子分支，新生的母簇加入到候选待分裂的母簇集中，继续开头所描述的过程。  

### 循环过程切入初始条件：  
初始：单一簇整个为新簇  
新生成的簇切入初始条件，内部二分割聚类，变成新生的母簇。初始的母簇是单一的。  
——母簇概念的明确：具备母子分支，又不分裂，才能称之为母簇。  
获取各母簇的母子分支的误差平方和降低量（在前一步新簇变为母簇时完成计算）  
从多个（非单一）母簇中选降低量最大的那一个分裂  
被分裂的母簇被内部已经二分割聚类生成的子簇（新生成的簇）替代，子簇变为新簇。  
重新循环

开始循环的第一步是新簇变母簇：  
新簇内部二分割聚类，共得到两簇母子分支。  
每次新的循环的标志事件是：  
新进两簇母簇入选待分裂的母簇集，可以区分出每次循环的不同候选集和不同的分裂结果。  



## 改进的二分聚类 选用的优化目标：母簇分裂的误差平方和降低量    
优化目标选择了母簇的母子分支的误差平方和降低量，而不是每个母簇分裂方案的总误差。    
类似决策树做分类的信息增益Gain，不同的是非监督方式，其实大道相通。  

簇内的误差平方和越小，样本相互越靠近，越纯洁，越容易划分到一类中来。    
误差平方和降低量越大，说明得到簇的过程（到达肘部elbow）越快，这一步迈得越大。  

SST=SSR+SSE 总离差平方和=簇间（组间）平方和（反映偏差）+簇内（组内）残差平方和（反映方差）   
取X=ε^即误差ε的估计量,根据方差性质E(X^2)=E(X)^2+D(X)，可以得知系统偏差和方差的影响。  

这两个理论为选取误差平方和为聚类分析指标提供了尺度上的把握。       
极端情况下：  
单一簇，误差平方和最大，为总离差；    
单样本簇，每个簇只有一个数据，误差平方和最小，为零。     
实际聚类情况：误差平方和达到应用要求的某一控制水平。  
误差平方和作为聚类的指示指标，是比较方便可行的。    

优化目标选择了母簇的母子分支的误差平方和降低量，而不是每个母簇分裂方案的总的误差平方和。  
这是为了程序选拔母簇有更加方便的针对指标，不需要关注总误差，只要具体到谁。  
母簇新生成时都通过二分聚类做了母子分支，并计算了母簇内的误差平方和降低量。     

此番改进不像标准的二分聚类程序，原有缺点是：   
每增加一个簇，要把各个母簇重复一次二分聚类，相当于重算一遍误差平方和降低量的工作量，而改进的程序仅对新生的母簇做二分聚类和误差统计。  
改进程序的存储空间保留了当前各本簇中心点、全体样本对各簇的从属关系、统计的本簇的误差项。   
增加了子簇中心点、各簇内样本对其各子簇的从属关系、子簇内的统计误差项。  

时空开销的权衡比较：         
空间最明显的增加就是子簇的从属关系，体量相当于全体样本对各簇的从属关系，增加了一倍的空间开销，但压缩了(K-2)的重复二分聚类和误差统计（最后一步K个聚类）的时间开销。    

改进的适用情况：   
越多的聚类个数，越需要压缩重复二分聚类和误差统计的时间开销，尽管二分聚类每个母簇的样本数也在减小。  
改进的误差平方和降低量优化方法更加适合。    



## 细节问题梳理（因果倒推，上因（准备+条件）下果，目标导向思维。）  
聚类细节上的因果关系清理 单一的簇变成母簇 生成母子分支，但未分裂  
初始到正常的母簇数量增长过程：

单一母簇直接分裂：    
                                      不用选择，就是她，没有选择的问题    
正常的分裂过程针对多个非单一的母簇   
                                      新生成的母簇，母簇之间确定：    
                                      谁是分裂者，数量至少是2个母簇。    
                                         对分裂的母簇二分割聚类生成子簇      
已有的各未分裂母子分支的误差平方和降低量    计算各母子分支的误差平方和降低量  
已有的母子分支（子簇未分裂出去）             更新的母子分支（子簇分裂出去）    
                   准备好各母子分支，误差平方和降低量     
从各母子分支中选拔误差平方和降低量最大的，聚类个数增加1，更新两子簇为母簇。    
聚类个数达到要求，停止聚类，不再分裂母簇。    




## 优化的kmeans数据字典  
术语概念空间：  
簇概念空间的划分为两维：本，母，子，新生；中心坐标，所属关系，误差平方项  

数据对象：列表、矩阵、数组；生命周期有初始化，使用，更新（生成）  
数据对象是簇等术语概念的载体。  
术语概念和关系需要根据程序设计来精化，不要搞大而全的覆盖。  

同SQL关系数据库一样要做E-R范式依赖关系的分解。  
为程序执行提供存储结构（安排一定的数据对象），也要为人的审查提供视图（连接查询）。  
E-R分解体现程序设计的精简要义。不能把全连接作为存储和执行单元。  
人机的设计和执行单元划分有所不同，人机各界：人的东西要聚合要包容，机的执行和存储要精当，组合连接要灵活，覆盖人的需求。    

本簇： 本簇（中心坐标，误差平方和SSE） CurrentClusters可分解的复合表     
currentCentroids表存放本簇中心点坐标√   
ClusterErrorSum表存放误差平方和SSE√
clusterAssment簇所属表，存放所属的簇的中心号，到簇中心的距离平方 √     
全体样本与所属簇的从属关系： 样本⁮点 → 簇中心 （所属簇的中心号，到簇中心的距离平方）      

母簇：本簇的扩展，程序中本簇的主要代表者，不另行命名 
母簇（Index：本簇中心号，子簇中心坐标,分裂后的残差，子簇划分带来的误差平方和降低量）     
其中，本簇中心号不用列出，直接排位区分，是索引。    
newCluster=[newClusterCentroids ID as Index->,subClusterCentroids,splitedError,ErrorDelta,subclusterAss] √ √
都可以分解独立成表 最终只取subClusterCentroids,subClusterAss（可导出subClusterErrorSum和splitedError）  
subClusterErrorSum=[np.sum(clusterAss[clusterAss[:,0]==i, 1]) for i in [0,1]]  
母簇内两种从属关系：     
全体样本与所属母簇的从属关系： 样本⁮点 → 母簇中心 （所属母簇的中心号，到母簇中心的距离平方）    
subclusterAss子簇所属表 √   
簇内样本与子簇的从属关系： 本簇内： 样本⁮点 → 子簇中心（所属子簇的中心号，到子簇中心的距离平方）  
#subClusterCentroids,splitedError,subClusterErrorSum,subClusterAss    
#=newCluster(ptsInCluster,k=2)#ptsInCluster为准母簇  

子簇： 子簇（中心坐标，误差平方和SSE） SubClusters可分解的复合表     
subClusterCentroids表存放子簇中心点坐标√   
subClusterErrorSum表存放各子簇的误差平方和SSE√ 身份只是中间表 用于分裂后，转正为ClusterErrorSum表存放误差平方和SSE    
ErrorDelta表 子簇划分带来的误差平方和降低量，二分割新簇得到的新生母簇时生成，用于分裂母簇选拔。    

辅助的历史记录表 在母簇分裂时更新 discenter表示被分裂的一代代母簇    
#存储有：中心点，含有的误差平方和，母子分支的误差平方和的降低量，分裂后的残差平方和，初始为np.inf   
disCenterIDs表，disCentroids表，disErrorSum表，disErrorDelta表，dissplitedError表  

新生簇（初始条件和循环结束条件）newCluster：   
新生簇中心表 newClusterCentroids 新生簇的中心点坐标，新生的是两个子簇，故为2组坐标的列表。     
新生簇的所属表 newClusterAssment 新生簇的误差平方和newClusterErrorSum 新生簇在当前簇集中的序号    Indexes_of_newClusterCentroids_In_currentCentroids 新生簇没有ErrorDelta表，因为还没有分裂    

数据对象的生命周期： 注意区分使用和生成（一般设空值）、初始化赋值（更新），以此为线索检查调试程序。    

## 技术小trick    
#按：注意：0,1两簇的编号如果直接按照needToSplit和len(currentCentroids) 更新      
    #因为newClusterAssment不能区分编号，会出现干扰。     
    #为保险起见，needToSplit先不做needToSplit更新，而是采用很大的数值    
    #大数值例如needToSplit+10000，替换该编号    
    #不出现混淆时，len(currentCentroids) 更新顺利完成后，needToSplit+10000改回needToSplit    
#注意对空簇的处置    
#按：重点细节 1号子簇要优先更新，以避免与0号子簇的needToSplit=1干扰情况出现：   
0号子簇如果先更新，有可能更新的值needToSplit恰好是1，在1号子簇更新时没有保护，也被更新掉！    
#获得剩余数据集的误差 nonSplitedError = np.sum(clusterAssment[np.nonzero( clusterAssment[:, 0].A != j)[0]][:, 1])    
不是从dataSet[np.nonzero( clusterAssment[:, 0].A != j)[0]]获取误差     
#newCentroids = centroids#.A #错误源，不能带.A，这是两个子簇的族，源代码有.A （变为数组）      
#分组严重的不均衡现象的原因？needToSplit = j的限制 初始化 needToSplit = 0 #提出来      
#注意：[0]可以把数据从单元素列表中取出    

## 鸣谢      
https://github.com/yoyoyohamapi/mit-ml 斯坦福机器学习完整 python 实现  
提供了原始学习素材，本篇参考了kmeans部分。    
对其标准的bi-kmeans做了验证性测试，并修改了错误，可以使用！    
然后，提出了优化的bi-kmeans    

## 建议
在jupyter notebook下使用runkMeans.ipynb文件   
## runkMeans.ipynb文件整体结构说明
代码正文共三个部分：  
1、加载共享的函数和库，增加了newCluster函数    
2、主调main ：作为测试口，提供同一数据集上的各个kmeans的测试比较    
改进的bi-kmeans用时更短。 
以下是比较的结果，可以亲试：  
#centroids, clusterAssment = kMeans(dataMat,6)   
    #6运行时间 0.2850019931793213  
    #kmeans.2运行时间  0.09799885749816895
#centroids, clusterAssment = biKmeans(dataMat, 6)   
    #6 运行时间 0.3880009651184082 #注意对空簇的处置        
    #2运行时间 0.13300228118896484
    centroids, clusterAssment = biKmeans_(dataMat,6)    
    #6运行时间 0.28999781608581543    
    #2运行时间 0.3789994716644287  

3、汇总各种方案，即各种bi-kmeans方法  
包括专门的精简代码，标准的bi-kmeans代码，本次改进的bi-kmeans代码  
原有的kmeans在共享的函数和库中，加载kmean.py即可  
1、2夹杂探求过程，供关注过程感兴趣者阅读。  


## 收获
1、 都是可运行的代码。原有素材的标准bi-kmeans代码调试发现错误已经修改。  
    成果可以拿来使用。拿来主义者到此止步。    
2、探索过程的分享。  
    逻辑结构化思维的修炼，目标导向训练    
    人机的理解，提炼数据结构。用人的理解做出符合机器擅长的事。  
    数据对象生命周期全链检查程序。  
    程序的实现，经过探索发现，是有章可循的。 
    功到自然成。

## 展望  
   希望陆续的推陈出新，扩展框架，贡献精品美餐，以飨读者。  




























