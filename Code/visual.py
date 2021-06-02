import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.manifold import TSNE

def visulize_item(predict_matrix, item_data):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.cla()
    predict_matrix = predict_matrix.T
    pca = decomposition.PCA(n_components=2)
    pca.fit(predict_matrix)
    X = pca.transform(predict_matrix)
    for i in range(5):
        mask = item_data[:, i + 1] == 1
        plt.scatter(X[mask, 0], X[mask, 1])

    plt.subplot(1, 2, 2)
    plt.cla()
    X = TSNE(n_components=2).fit_transform(predict_matrix)
    for i in range(5):
        mask = item_data[:, i + 1] == 1
        plt.scatter(X[mask, 0], X[mask, 1])
    plt.show()
