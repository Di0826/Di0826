import numpy as np
import data_reader
import svd_cf
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.manifold import TSNE


def recommendation():
    user_data = data_reader.read_user()  # 943 users
    item_data = data_reader.read_item()  # 1682 items(movies)
    latent_variable_SVD = 8  # used by svd
    learning_rate_SVD = 0.01
    maximum_epoches = 1
    regularization = 0.005

    for test_num in range(1, 2):
        print("Test" + str(test_num) + ":")

        train = data_reader.read_rating_train(test_num)
        test = data_reader.read_rating_test(test_num)

        train_matrix = np.zeros((user_data.shape[0], item_data.shape[0]))
        # shape = num_user * num_items
        train_matrix[train[:, 0], train[:, 1]] = train[:, 2]
        predict_matrix = svd_cf.svd_collaborative_filtering(train, train_matrix, latent_variable_SVD,
                                                            learning_rate_SVD, maximum_epoches, regularization)
        # using svd to fill the matrix
        print("RMSE loss of " + "test " + str(test_num) + " is equal to " + str(svd_cf.RMSE_loss(predict_matrix, test)))

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.cla()
        predict_matrix = predict_matrix.T
        pca = decomposition.PCA(n_components=2)
        pca.fit(predict_matrix)
        X = pca.transform(predict_matrix)
        for i in range(5):
            mask = item_data[:, i+1] == 1
            plt.scatter(X[mask, 0], X[mask, 1])

        plt.subplot(1, 2, 2)
        plt.cla()
        X = TSNE(n_components=2).fit_transform(predict_matrix)
        for i in range(5):
            mask = item_data[:, i+1] == 1
            plt.scatter(X[mask, 0], X[mask, 1])
        plt.show()


if __name__ == '__main__':
    recommendation()

