from matplotlib import pyplot as plt
import pandas as pd
import pymc as pm
import numpy as np
import requests
import arviz as az


import warnings

warnings.filterwarnings("ignore", message="The figure layout has changed to tight")


class BeyesModel(pm.Model):
    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_coord(name="seed_num", values=[f"ID{i+1:02}" for i in range(len(data.y.values))])

        # βの事前分布をN(0, 100)の正規分布で設定(無情報事前分布)
        beta = pm.Normal("beta", mu=0, sigma=100)

        # 超パラメータsの(超)事前分布をU(0, 10000)の連続一様分布で設定(無情報事前分布)
        s = pm.Uniform("s", lower=0, upper=10000)

        # パラメータrの事前分布をN(0, s)の正規分布で設定(階層事前分布)
        r = pm.Normal("r", mu=0, sigma=s, dims="seed_num")

        # ロジットリンク関数を設定し、二項分布で推定する
        pm.Binomial("y", n=8, p=pm.invlogit(beta + r), observed=data.y.values)


def main() -> None:
    # 著者サイトからダウンロード
    response = requests.get("http://hosho.ees.hokudai.ac.jp/~kubo/stat/iwanamibook/fig/hbm/data7a.csv")
    with open("data7a.csv", "wb") as f:
        f.write(response.content)
        f.close()
    data = pd.read_csv("data7a.csv")
    print(data)

    model = BeyesModel(data)
    gv = pm.model_to_graphviz(model)
    gv.render(outfile="model.png")

    seed = 0
    rng = np.random.default_rng(seed)
    idata = pm.sample(
        draws=2000,
        tune=1000,
        random_seed=rng,
        model=model,
    )

    df_summary = az.summary(
        idata,
        kind="stats",
        hdi_prob=0.95,
    )
    df_summary.to_csv("summary.csv")

    az.plot_forest(
        idata,
        combined=True,
        hdi_prob=0.95,
        figsize=(8, 6),
    )
    plt.savefig("forestplot.png")
