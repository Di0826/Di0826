import matplotlib.pyplot as plt
from sklearn import decomposition
import numpy as np
from sklearn.cluster import KMeans

def visulize_item(predict_matrix, item_data):
    '''
    

    Parameters
    ----------
    predict_matrix : np arrays
        the predict_matrix with shape 984*1682
        each element of the data represent the rating for a perspon
    item_data : np arrays
        the detail information for 
        a moive including id title class data etc.

    Returns
    -------
    a image that visualized random 100 moview 

    '''
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.cla()
    
    
    predict_matrix = predict_matrix.T
    
    index = np.random.randint(1682, size=100)
    
    predict_matrix = predict_matrix[index]
    
    pca = decomposition.PCA(n_components=2)
    pca.fit(predict_matrix)
    X = pca.transform(predict_matrix)
    
    
    Y = np.zeros((100,2))
    Y[:, 0] = X[:,0]
    Y[:, 1] = X[:,1]
    

    y_pred = KMeans(n_clusters=5, random_state=9).fit_predict(Y)
    plt.scatter(Y[:, 0], Y[:, 1],c = y_pred)
    
    indmax = []
    countmax = 0
    for i in range(100):
        if y_pred[i] == 0:
            indmax.append(i) 
            countmax += 1
            
    indmin = []   
    countmin = 0
    for i in range(100):
        if y_pred[i] == 4:
            indmin.append(i) 
            countmin += 1
     
    ind2 = []        
    count2 = 0
    for i in range(100):
        if y_pred[i] == 1:
            ind2.append(i) 
            count2 += 1    
            
    ind3 = []        
    count3 = 0
    for i in range(100):
        if y_pred[i] == 2:
            ind3.append(i) 
            count3 += 1  

    ind4 = []        
    count4 = 0
    for i in range(100):
        if y_pred[i] == 3:
            ind4.append(i) 
            count4 += 1              
            
            
    maxdata = 0
    for item in indmax:
        maxdata = maxdata + sum(predict_matrix[item]) / 943
    mindata = 0  
    
    for item in indmin:
        mindata = mindata + sum(predict_matrix[item]) / 943        
    
    data2 = 0
    for item in ind2:
        data2 = data2 + sum(predict_matrix[item]) / 943   
        
    data3 = 0
    for item in ind3:
        data3 = data3 + sum(predict_matrix[item]) / 943     
    
    data4 = 0        
    for item in ind4:
        data4 = data4 + sum(predict_matrix[item]) / 943     
    
    
    
    
    print(  maxdata /  countmax)
    print(  mindata /  countmin)
    print(data2 / count2)
    print(data3 / count3)
    print(data4 / count4)

    
        

    
    

    


