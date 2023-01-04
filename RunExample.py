import matplotlib.pyplot as plt
import numpy as np
from LoadData import load_rating_data, spilt_rating_dat
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF

if __name__ == "__main__":
    # file_path = "data/ml-100k/u.data"
    file_path = "data/Video_Games_5.data"
    pmf = PMF()
    pmf.set_params({"num_feat": 10, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 50, "num_batches": 100,
                    "batch_size": 1000})
    ratings = load_rating_data(file_path)
    print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
    train, test = train_test_split(ratings, test_size=0.2)  # spilt_rating_dat(ratings)
    pmf.fit(train, test)

    # Check performance by plotting train and test errors
    plt.plot(range(pmf.maxepoch), pmf.mse_train, marker='o', label='Training Data')
    plt.plot(range(pmf.maxepoch), pmf.mse_test, marker='v', label='Test Data')
    plt.title('The '+file_path+' Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid()
    plt.show()
    print("precision_acc,recall_acc:" + str(pmf.topK(test)))
