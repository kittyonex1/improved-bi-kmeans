# %load test_normal_kmeans.py
# test_normal_kmeans.py
import time as tm
import kmeans
import numpy as np
import matplotlib.pyplot as plt
#按：在Notepad++中运行python脚本 cmd /k D:\ProgramData\Anaconda2\envs\py3\python.exe "$(FULL_CURRENT_PATH)" & PAUSE & EXIT
if __name__ == "__main__":
    s=tm.time()
    dataMat = np.mat(loadDataSet('data/testSet.txt'))#kmeans.
    centroids, clusterAssment = kMeans(dataMat,6) 
    #6运行时间 0.2689971923828125
    #6运行时间 0.2850019931793213
    #6运行时间 0.2919943332672119
    #kmeans.2运行时间  0.08598971366882324
#     centroids, clusterAssment = biKmeans(dataMat, 6)
    #6运行时间 0.39299678802490234
    #6运行时间 0.42099523544311523
    #6运行时间 0.3880009651184082 
    #2运行时间 0.072998046875
#     centroids, clusterAssment = biKmeans_(dataMat,6)  
    #6运行时间 0.3679966926574707  
    #6运行时间 0.3579981327056885
    #6运行时间 0.28999781608581543  
    #2运行时间 0.11799740791320801
    clusterCount = np.shape(centroids)[0]
    m = np.shape(dataMat)[0]
    print('运行时间',tm.time()-s)
    # 绘制散点图
    patterns = ['o', 'D', '^', 's','*','P','H','x']   #discent消失的中心备用'v'倒三角
    colors = ['b', 'g', 'y', 'black','c','r','m','w']
    #补充：颜色参数：
#b--blue    c--cyan    g--green  k--black
#m--magenta r--red     w--white  y--yellow
    fig = plt.figure()
    title = 'kmeans with k='+str(clusterCount)
    ax = fig.add_subplot(111, title=title)
    for k in range(clusterCount):
        # 绘制聚类中心    #color='r修改，以便辨认
        #空簇处理放在biKmeans等中
        ax.scatter(centroids[k, 0], centroids[k, 1], color=str(colors[k]), marker='+', linewidth=20)
        for i in range(m):
            # 绘制属于该聚类中心的样本
            ptsInCluster = dataMat[np.nonzero(clusterAssment[:, 0].A==k)[0]]
            ax.scatter(ptsInCluster[:, 0].flatten().A[0], ptsInCluster[:, 1].flatten().A[0], marker=patterns[k], color=colors[k])
    plt.show()