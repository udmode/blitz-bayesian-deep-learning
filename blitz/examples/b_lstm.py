from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@variational_estimator
class NN(nn.Module):
    """
    ベイジアンLSTMを使用したニューラルネットワーク
    時系列データのパターンを学習し、未来の株価を予測するために使用される
    """

    def __init__(self):
        """
        ニューラルネットワークのレイヤーを定義する
        BayesianLSTMは、LSTM（Long Short-Term Memory）は、時系列データのパターンを学習するために使用される
        nn.Linearは、全結合レイヤー（線形レイヤー）であり、ニューラルネットワークの各レイヤーが前のレイヤーと全結合していることを意味する
        """
        super(NN, self).__init__()
        self.lstm_1 = BayesianLSTM(1, 10, prior_sigma_1=1, prior_pi=1, posterior_rho_init=-3.0)
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        """
        ニューラルネットワークの順伝播を定義する
        つまり、入力データがどのようにネットワークを通過し、出力を生成するかを定義
        まずBayesianLSTMレイヤーを通過し、その出力が全結合レイヤーに渡され、最終的な出力が生成される
        """
        x_, _ = self.lstm_1(x)

        # gathering only the latent end-of-sequence for the linear layer
        x_ = x_[:, -1, :]
        x_ = self.linear(x_)
        return x_


class PredictionByBayesianLSTM:

    def __init__(self):
        self.file_path = "data/"
        self.input_file = "oanda_db_USD_JPY_M1_20240503133142.csv"
        self.df = pd.read_csv(f"{self.file_path}{self.input_file}")
        print(self.df.head())

    def create_timestamps_ds(self, series, timestep_size):
        print(f"series: {series}")
        print(f"timestep_size: {timestep_size}")
        """
        時系列データをモデルが処理できる形式に変換する
        """
        time_stamps = []
        labels = []

        # collectionsモジュールのdeque（デック）というデータ構造を使用する
        # dequeは、両端で要素の追加や削除が高速に行えるリスト型のデータ構造です
        aux_deque = deque(maxlen=timestep_size)
        print(f"aux_deque: {aux_deque}")
        # starting the timestep deque
        for i in range(timestep_size):
            aux_deque.append(0)
            print(f"range(timestep_size): {i}")
        # feed the timestamps list
        for i in range(len(series) - 1):
            aux_deque.append(series[i])
            time_stamps.append(list(aux_deque))
            print(f"range(len(series) - 1): {i}")
        # feed the labels lsit
        for i in range(len(series) - 1):
            labels.append(series[i + 1])
            print(f"range(len(series) - 1): {i}")

        assert len(time_stamps) == len(labels), "Something went wrong"

        # torch-tensoring it 浮動小数点数のテンソルに変換
        # テンソルは、多次元配列のようなもので、PyTorchではニューラルネットワークの入力データやパラメータ、出力データを表現するために使用される
        features = torch.tensor(time_stamps[timestep_size:]).float()
        labels = torch.tensor(labels[timestep_size:]).float()

        return features, labels

    def pred_stock_future(self, X_test, future_length, sample_nbr=10):
        """
        与えられたテストデータ X_test を用いて、未来の株価を予測するためのものです。
        ベイジアンニューラルネットワークのモデルを使用して、指定された未来の日数 future_length にわたる株価の予測値を生成します。
        """
        global window_size
        global X_train
        global Xs
        global scaler

        # 予測値を格納するための空のリストを作成
        preds_test = []
        # 予測を開始する最初のデータポイントを取得
        # テストデータの最初のウィンドウを取得し、それをdeque（両端キュー）に変換します。
        # dequeは、両端で要素の追加や削除が高速に行えるリスト型のデータ構造です。
        test_begin = X_test[0:1, :, :]
        test_deque = deque(test_begin[0, :, 0].tolist(), maxlen=window_size)
        # 予測値のインデックスをdf["Date"] のような日付型を生成
        idx_pred = self.df["Date"][len(X_train) + window_size:].values
        # idx_pred = np.arange(len(X_train), len(Xs))

        # テストデータの各ウィンドウに対してループ
        for i in range(len(X_test)):
            # テストデータのウィンドウをモデルに供給し、未来の株価を予測
            # 現在のウィンドウをニューラルネットワークの入力として使用できる形式に変換します。
            as_net_input = torch.tensor(test_deque).unsqueeze(0).unsqueeze(2)
            # 現在のウィンドウに対する予測値を生成します。sample_nbr 回繰り返され、それぞれの予測値がリスト pred に格納されます。
            pred = [net(as_net_input).cpu().item() for i in range(sample_nbr)]
            # 生成した予測値の平均をdequeに追加し、予測値のリスト全体を preds_test に追加します。
            test_deque.append(torch.tensor(pred).mean().cpu().item())
            preds_test.append(pred)

            # 指定された未来の日数ごとに、dequeとテストデータのウィンドウを更新します。
            if i % future_length == 0:
                # our inptus become the i index of our X_test
                # That tweak just helps us with shape issues
                test_begin = X_test[i:i + 1, :, :]
                test_deque = deque(test_begin[0, :, 0].tolist(), maxlen=window_size)

        # preds_test = np.array(preds_test).reshape(-1, 1)
        # preds_test_unscaled = scaler.inverse_transform(preds_test)

        return idx_pred, preds_test

    def get_confidence_intervals(preds_test, ci_multiplier):
        """
        予測結果の信頼区間を計算するためのものです。具体的には、予測値の平均と標準偏差を用いて、信頼区間の上限と下限を計算します。
        """
        global scaler
        # 予測値をPyTorchのテンソルに変換
        preds_test = torch.tensor(preds_test)
        # 予測値の平均を計算
        pred_mean = preds_test.mean(1)
        # 予測値の標準偏差を計算。テンソルからnumpy配列に変換。
        pred_std = preds_test.std(1).detach().cpu().numpy()
        # 予測値の標準偏差をPyTorchのテンソルに変換
        pred_std = torch.tensor((pred_std))
        # print(pred_std)

        # 信頼区間の上限と下限を計算します。ci_multiplierは信頼区間の幅を制御するパラメータ
        upper_bound = pred_mean + (pred_std * ci_multiplier)
        lower_bound = pred_mean - (pred_std * ci_multiplier)

        # 平均値、上限、下限を元のスケールに戻します。これは、モデルの入力データがスケーリング（正規化）されていた場合に必要です。
        pred_mean_final = pred_mean.unsqueeze(1).detach().cpu().numpy()
        pred_mean_unscaled = scaler.inverse_transform(pred_mean_final)

        upper_bound_unscaled = upper_bound.unsqueeze(1).detach().cpu().numpy()
        upper_bound_unscaled = scaler.inverse_transform(upper_bound_unscaled)

        lower_bound_unscaled = lower_bound.unsqueeze(1).detach().cpu().numpy()
        lower_bound_unscaled = scaler.inverse_transform(lower_bound_unscaled)

        return pred_mean_unscaled, upper_bound_unscaled, lower_bound_unscaled

    def exec(self):
        close_prices = self.df["Close"]
        # 特徴量のスケーリングを行うためのツールで、特に機械学習モデルを訓練する前にデータを前処理する際に使用される
        scaler = StandardScaler()

        close_prices_arr = np.array(close_prices).reshape(-1, 1)
        # スケーリング
        close_prices = scaler.fit_transform(close_prices_arr)

        window_size = 21
        Xs, ys = self.create_timestamps_ds(close_prices, window_size)

        X_train, X_test, y_train, y_test = train_test_split(
            Xs,
            ys,
            test_size=.25,
            random_state=42,
            shuffle=False)

        # 訓練データ（X_trainとy_train）をデータセットに変換
        ds = torch.utils.data.TensorDataset(X_train, y_train)
        # 訓練データをバッチ処理するためのデータローダーを作成
        # バッチサイズは一度にネットワークに供給されるデータの数（バッチサイズ）を指定
        dataloader_train = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
        print(f"Train size: {len(X_train)} Test size: {len(X_test)}")

        net = NN()
        # 損失関数を定義する。平均二乗誤差（MSE）を使用する
        # 損失関数は、モデルの予測が実際のラベルからどれだけ離れているかを測定する
        # 訓練中には、この損失を最小化するようにモデルのパラメータが調整される
        criterion = nn.MSELoss()
        # 最適化アルゴリズムを定義する。Adamアルゴリズムを使用する
        # Adamは、確率的勾配降下法（SGD）の一種であり、モデルのパラメータを最適化するために使用される
        # lr=0.001は、学習率を指定する。学習率は、モデルのパラメータを更新するステップの大きさを制御する
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        iteration = 0
        # エポックのループです。エポックとは、訓練データ全体を一度通過することを指す
        for epoch in range(10):
            # バッチのループ。バッチとは、訓練データを一部分ずつ処理するための小さなデータの塊
            for i, (datapoints, labels) in enumerate(dataloader_train):
                # 最適化アルゴリズムの勾配をゼロにリセット
                optimizer.zero_grad()

                # モデルの損失（エラー）を計算します。この損失は、モデルの予測と実際のラベルとの差を表す
                # sample_elboは、ベイジアンニューラルネットワークの損失を計算するための関数
                loss = net.sample_elbo(
                    inputs=datapoints,
                    labels=labels,
                    criterion=criterion,
                    sample_nbr=3,
                    complexity_cost_weight=1 / X_train.shape[0])
                # 損失に対する各パラメータの勾配を計算する。パラメータを更新するために必要
                loss.backward()
                # 計算された勾配を使用して、モデルのパラメータを更新する
                optimizer.step()

                # 250回のイテレーションごとに、テストデータに対する損失を計算し、表示する
                iteration += 1
                if iteration % 250 == 0:
                    preds_test = net(X_test)[:, 0].unsqueeze(1)
                    loss_test = criterion(preds_test, y_test)
                    print("Iteration: {} Val-loss: {:.4f}".format(str(iteration), loss_test))
        # %%
        # pred_unscaled
        original = self.df["Close"][1:][window_size:]  # データフレームdfから「Close」列を取得し、最初の要素とwindow_size以降の要素をスキップ
        df_pred = pd.DataFrame(original)
        df_pred["Date"] = self.df.Date
        self.df["Date"] = pd.to_datetime(df_pred["Date"])  # 「Date」列を日付型（datetime）に変換
        df_pred = df_pred.reset_index()  # インデックスを0から始まる連番になるようにリセット

        # future_length=7
        future_length = 3
        # sample_nbr = 4
        sample_nbr = 10
        # ci_multiplier = 5
        ci_multiplier = 2
        idx_pred, preds_test = self.pred_stock_future(X_test, future_length, sample_nbr)
        pred_mean_unscaled, upper_bound_unscaled, lower_bound_unscaled = self.get_confidence_intervals(
            preds_test, ci_multiplier)

        # 実際の株価（df.Close）の最後の部分（予測結果と同じ長さ）を取得し、それを2次元配列に変換しています。
        # y = np.array(df.close[-750:]).reshape(-1, 1)
        y = np.array(self.df.Close[-len(upper_bound_unscaled):]).reshape(-1, 1)
        # 予測結果の信頼区間の上限upper_bound_unscaled と下限 lower_bound_unscaled を比較し、それらが実際のデータ y の範囲内にあるかどうかを確認します。
        under_upper = upper_bound_unscaled > y
        over_lower = lower_bound_unscaled < y
        # 上記の2つのブール値配列が等しいかどうかを判定しています。これは、実際の株価が信頼区間内に収まっている場合にTrueとなります。
        total = (under_upper == over_lower)

        print("{} our predictions are in our confidence interval".format(np.mean(total)))

        # Date、original, pred_mean_unscaled, upper_bound_unscaled, lower_bound_unscaledの値をCSVに出力する
        # 保存先はdata/prediction.csvとする
        # ヘッダーはDate、original、pred_mean_unscaled、upper_bound_unscaled、lower_bound_unscaledとする
        # df_pred.columns = ["Date", "original", "pred_mean_unscaled", "upper_bound_unscaled", "lower_bound_unscaled"]
        # df_pred["Date"] = df.Date
        # df_pred["original"] = original
        df_new = pd.DataFrame()
        df_new.index = idx_pred
        df_new["real"] = df_pred.Close[-len(pred_mean_unscaled):]
        df_new["pred_mean_unscaled"] = pred_mean_unscaled
        df_new["upper_bound_unscaled"] = upper_bound_unscaled[:, 0]
        df_new["lower_bound_unscaled"] = lower_bound_unscaled[:, 0]

        # 現時刻を取得して、YYYY_MM_DD-HH_mm_SSのフォーマットにする
        import datetime

        # CSVファイルに出力
        df_new.to_csv(
            f"data/prediction_{self.input_file.split('.')[0]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")


if __name__ == "__main__":
    pbl = PredictionByBayesianLSTM()
    pbl.exec()
