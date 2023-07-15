import pandas as pd
from sklearn.datasets import fetch_california_housing
import seaborn as sns
from matplotlib import pyplot as plt
from statistics import mean, median, variance, stdev
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from src.utils import show_measurements, test_shapiro_wilk, test_kolmogorov_smirnov, test_qqplot, collinearity_check
import warnings

warnings.filterwarnings("ignore", message="The figure layout has changed to tight")


def main() -> None:
    boston = fetch_california_housing()
    y = boston.target
    test_qqplot(y)
    sns.displot(y)
    plt.show()

    y_boxcox, lambda_boxcox = boxcox(y)  # boxcox transformation (Xc should follow bormal destribution)
    test_kolmogorov_smirnov(y_boxcox)
    test_qqplot(y_boxcox)
    sns.displot(y_boxcox)
    plt.show()

    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    print(df.head())

    df_std = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)  # standardization
    collinearity_check(df_std)
