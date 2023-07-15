from typing import List
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from statistics import mean, median, variance, stdev
from scipy.stats import shapiro, ks_1samp, norm, probplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show_measurements(target_varuables: List[float]) -> None:
    print(f"n: {len(target_varuables)}")
    print(f"mean: {mean(target_varuables)}")
    print(f"median: {median(target_varuables)}")
    print(f"variance: {variance(target_varuables)}")
    print(f"stdev: {stdev(target_varuables)}")


def test_shapiro_wilk(target_varuables: List[float]) -> None:
    """
    Shapiro-Wilk Test to check if the distribution is normal
    to be applied to a small sample size (n <= 50)
    p < 0.05: reject the null hypothesis (not normal)
    """
    stat, p = shapiro(target_varuables)
    print(f"Shapiro-Wilk Test: stat={stat}, p={p}")


def test_kolmogorov_smirnov(target_varuables: List[float]) -> None:
    """
    kolmoogorov-smirnov Test to check if the distribution is normal
    to be applied to a large sample size (n > 50)
    p < 0.05: reject the null hypothesis (not normal)
    """
    stat, p = ks_1samp(target_varuables, norm.cdf)
    print(f"Kolmogorov-Smirnov Test: stat={stat}, p={p}")


def test_qqplot(target_varuables: List[float]) -> None:
    """
    QQ plot to check if the distribution is normal
    """
    probplot(target_varuables, dist="norm", plot=plt)


def collinearity_check(Xc_df: pd.DataFrame, *, model=None, alpha: float = 1.0, emph: bool = True) -> None:
    """
    check collinearity of explanatory variables
    R2 > 0.9: high collinearity
    R2 > 0.8: moderate collinearity
    R2 > 0.7: weak collinearity

    Parameters
    ----------
    Xc : pd.DataFrame(->np.ndarray(m, n))
        Input data (each data stored in a row). It should be standardized beforehands.
    model
        Regression model (default Ridge).
    alpha : float
        Hyper parameter of Ridge (default 1.0),
        ignored if model is not None.
    emph : bool
        Emphasize the result by R2 score or not.

    Returns
    -------
    rc : np.ndarray(n, n)
        Regression coefficient, emphasized by R2 score if emph is True.
    scores : np.ndarray(n)
        R2 scores.
    """

    Xc = Xc_df.to_numpy(copy=True)
    header = Xc_df.columns

    if model is None:
        model = Ridge(alpha=alpha)

    m, n = Xc.shape
    if n < 2:
        raise ValueError()

    # 戻り値
    rc = np.empty((n, n))  # 回帰係数(regression coefficient)
    r2_scores = np.empty(n)  # R2スコア(決定係数)

    X = np.copy(Xc)
    for i in range(n):
        y = np.copy(X[:, i])  # 自分を他の変数で回帰させる
        X[:, i] = 1  # 自分は多重共線性の定数項に対応させる

        model.fit(X, y)
        y_calc = model.predict(X)

        score = r2_score(y, y_calc)
        if score < 0:
            # R2 スコアが 0 以下なら線形性なしとみなす
            r2_scores[i] = 0
            rc[i] = 0
        else:
            r2_scores[i] = score
            if emph:
                # 係数が大きくても R2 スコアが 0 に近ければ 0 になるように加工
                rc[i] = model.coef_ * score
            else:
                rc[i] = model.coef_

        X[:, i] = y

    for i in range(n):
        print(f"{i}({header[i]}): rc values={rc[i]}, r2 value={r2_scores[i]}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(rc), vmin=-1, vmax=1, cmap="bwr", cbar=True)
    plt.ylabel("target variable", fontsize=16)
    plt.xlabel("used variables to explain the target", fontsize=16)
    plt.show()
