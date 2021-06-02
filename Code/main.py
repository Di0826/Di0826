import numpy as np
import data_reader
import svd_cf
import visual
import neuralnet

def recommendation():
    user_data = data_reader.read_user()
    # 943 users, 2 features, gender and age
    item_data = data_reader.read_item()
    # 1682 items(movies), 20 features, the labels (category) and the year of release date
    latent_variable_SVD = 8  # recommended value 8
    learning_rate_SVD = 0.01  # recommended value 0.01
    maximum_epoches_SVD = 25  # recommended value 25
    regularization_SVD = 0.005  # recommended value 0.005

    for test_num in range(1, 2):
        print("Test" + str(test_num) + ":")

        train = data_reader.read_rating_train(test_num)
        test = data_reader.read_rating_test(test_num)

        # svd
        predict_matrix = svd_cf.svd_collaborative_filtering(train, user_data, item_data, latent_variable_SVD,
                                                            learning_rate_SVD, maximum_epoches_SVD, regularization_SVD)
        print("RMSE loss of SVD CF on test %d is equal to %.6f" %(test_num, svd_cf.RMSE_loss(predict_matrix, test)))
        # visual.visulize_item(predict_matrix, item_data)

        # neural net
        predict_matrix = neuralnet.neuralnet_predictor(train, user_data, item_data)
        print("RMSE loss of Neural Net on test %d is equal to %.6f" %(test_num, svd_cf.RMSE_loss(predict_matrix, test)))
        # visual.visulize_item(predict_matrix, item_data)

def discrete(data):
    data = np.where(data < 1.5, 1, data)
    data = np.where((data >= 1.5) & (data < 2.5), 2, data)
    data = np.where((data >= 2.5) & (data < 3.5), 3, data)
    data = np.where((data >= 3.5) & (data < 4.5), 4, data)
    data = np.where(data > 4.5, 5, data)
    return data


if __name__ == '__main__':
    recommendation()

