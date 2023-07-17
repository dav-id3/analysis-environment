from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from src.utils import BeyesLinearRegression
import warnings

warnings.filterwarnings("ignore", message="The figure layout has changed to tight")


def main() -> None:
    # サンプルデータセットからtrain data, test dataを作成する
    # # サンプルデータ（ガウスノイズを付加）
    # x_sample = np.arange(0, 1.01, 0.02)
    # y_sample = np.sin(x_sample * np.pi * 2) + np.random.normal(loc=0.0, scale=0.1, size=np.size(x_sample))
    # x_train, x_test, y_train, y_test = train_test_split(x_sample, y_sample, test_size=0.2, random_state=0, shuffle=True)
    # # テストデータをxに関する昇順でソート
    # x_test, y_test = zip(*sorted(list(zip(x_test, y_test)), key=lambda x: x[0]))
    # x_test = np.array(x_test)
    # y_test = np.array(y_test)

    # train data, test data（ガウスノイズを付加）
    x_train = np.arange(0, 1.01, 0.05)
    y_train = np.sin(x_train * np.pi * 2) + np.random.normal(loc=0.0, scale=0.1, size=np.size(x_train))
    x_test = np.arange(0, 1.01, 0.002)
    y_test = np.sin(x_test * np.pi * 2)

    # ベイズ線形回帰
    bayes_gaussian = BeyesLinearRegression(x_train, y_train, phi_type="gaussian", sigma=0.1, M=9)
    bayes_gaussian.calc_posterior()
    bayes_gaussian.predict_and_visualization(x_test, y_test)
