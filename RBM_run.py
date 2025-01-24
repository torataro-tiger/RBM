import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class RBM(nn.Module):
    def __init__(
        self,
        n_visible_nodes: int = 28 * 28,
        m_hidden_nodes: int = 128,
        epoch: int = 10,
        lr: float = 0.01,
    ):
        """
        Args:
            n_visible_nodes (int, optional): 可視層のノード数. Defaults to 28*28.
            m_hidden_nodes (int, optional): 隠れ層のノード数. Defaults to 128.
            epoch (int, optional): エポック数. Defaults to 10.
            lr (float, optional): 学習率. Defaults to 0.01.
        """
        super().__init__()
        self.m_hidden_nodes = m_hidden_nodes
        self.n_visible_nodes = n_visible_nodes
        self.epoch = epoch
        self.lr = lr
        self.W = nn.Parameter(torch.zeros(m_hidden_nodes, n_visible_nodes))
        self.b = nn.Parameter(torch.zeros(n_visible_nodes))
        self.c = nn.Parameter(torch.zeros(m_hidden_nodes))

    def sample_hidden_vec(self, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        可視層の状態から隠れ層の状態をサンプリングする。

        Args:
            v (Tensor): 可視層の状態

        Returns:
            tuple[Tensor, Tensor]: p(h_{j}=1)の確率とサンプリングした隠れ層の状態
        """
        _lambda = torch.matmul(v, self.W.t()) + self.c
        prob_h = torch.sigmoid(_lambda)
        h_sample = torch.bernoulli(prob_h)
        return prob_h, h_sample

    def sample_visible_vec(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """
        隠れ層の状態から可視層の状態をサンプリングする。

        Args:
            h (Tensor): 隠れ層の状態

        Returns:
            tuple[Tensor, Tensor]: p(v_{i}=1)の確率とサンプリングした可視層の状態
        """
        _lambda = torch.matmul(h, self.W) + self.b
        prob_v = torch.sigmoid(_lambda)
        v_sample = torch.bernoulli(prob_v)
        return prob_v, v_sample

    def cd_method(self, v_0: Tensor, N: int) -> None:
        """
        1回サンプリングのCD法と勾配計算

        Args:
            v_0 (Tensor): 観測した生データ
            N (int): バッチ内の総データ数
        """
        # 1回サンプリングののCD法
        prob_h_0, h_0 = self.sample_hidden_vec(v_0)
        _, v_1 = self.sample_visible_vec(h_0)
        _, h_1 = self.sample_hidden_vec(v_1)

        # 勾配算出
        self.W.grad = (torch.matmul(h_1.t(), v_1) - torch.matmul(prob_h_0.t(), v_0)) / N
        self.b.grad = torch.sum(v_1 - v_0, dim=0) / N
        self.c.grad = torch.sum(h_1 - prob_h_0, dim=0) / N

    @torch.no_grad()
    def update_parameters(self) -> None:
        """
        パラメータ更新
        """
        self.W -= self.lr * self.W.grad  # type: ignore
        self.b -= self.lr * self.b.grad  # type: ignore
        self.c -= self.lr * self.c.grad  # type: ignore

    def RMSE(self, v_recon: Tensor, v: Tensor) -> Tensor:
        """
        二乗平均平方根誤差(RMSE)計算

        Args:
            v_recon (Tensor): 再構成後のデータ
            v (Tensor): 生データ

        Returns:
            Tensor: RMSE
        """
        RMSE = torch.sqrt(torch.mean((v_recon - v) ** 2))
        return RMSE

    def forward(self, v: Tensor) -> Tensor:
        """
        学習したモデルをもとにデータを再構成する。

        Args:
            v (Tensor): 再構成したい生データ

        Returns:
            Tensor: 再構成データ
        """
        _, h = self.sample_hidden_vec(v)
        _, v_recon = self.sample_visible_vec(h)
        return v_recon

    def fit(self, train_loader: DataLoader) -> list[float]:
        """制限ボルツマンマシンの学習

        Args:
            train_loader (DataLoader): バッチデータ生成用のデータローダ

        Returns:
            list[float]: 各エポックごとのRMSE
        """
        RMSE_list = []
        for i in range(self.epoch):
            RMSE = 0
            N_batch = 0
            for batch in train_loader:
                self.zero_grad()
                batch_reshaped = batch[0].view(-1, self.n_visible_nodes)
                N = batch_reshaped.size()[0]  # バッチ内の総データ数
                self.cd_method(v_0=batch_reshaped, N=N)
                self.update_parameters()

                # 損失計算
                v_recon = self(batch_reshaped)
                RMSE += self.RMSE(v_recon=v_recon, v=batch_reshaped).item()
                N_batch += 1
            RMSE_ave = RMSE / N_batch
            RMSE_list.append(RMSE_ave)
            print(f"epoch: {i+1}, RMSE={RMSE_ave:.4f}")
        return RMSE_list


if __name__ == "__main__":

    # コマンドライン引数関連
    parser = argparse.ArgumentParser(description="制限ボルツマンマシンの学習とその結果を表示するコードです。")

    parser.add_argument("--batch_size", type=int, default=32, help="バッチサイズです。(デフォルト値: %(default)s)")
    parser.add_argument(
        "--result_show_num",
        type=int,
        default=8,
        help="訓練結果のデータの表示数です。指定した数のテストデータと再構成結果が表示されます。(デフォルト値: %(default)s)",
    )
    parser.add_argument(
        "--download_MNIST",
        action="store_true",
        help="MNIST画像をダウンロードするかしないかを決めてください。「--download_MNIST」を指定するだけでダウンロードが開始します。",
    )
    parser.add_argument(
        "--m_hidden_nodes", type=int, default=128, help="隠れ層のノード数です。(デフォルト値: %(default)s)"
    )
    parser.add_argument("--epoch", type=int, default=10, help="学習時のエポック数です。(デフォルト値: %(default)s)")
    parser.add_argument("--lr", type=float, default=0.01, help="学習率です。(デフォルト値: %(default)s)")

    args = parser.parse_args()

    batch_size = args.batch_size
    result_show_num = args.result_show_num
    download_MNIST = args.download_MNIST
    n_visible_nodes = 28 * 28
    m_hidden_nodes = args.m_hidden_nodes
    epoch = args.epoch
    lr = args.lr

    print(
        f"""
    # パラメータ設定
    - バッチサイズ: {batch_size}
    - 可視層ノード数: {n_visible_nodes}
    - 隠れ層ノード数: {m_hidden_nodes}
    - エポック数: {epoch}
    - 学習率: {lr}
    """
    )

    # 訓練データ・テストデータ読み込み
    transform = transforms.Compose([transforms.ToTensor(), lambda x: (x > 0.5).float()])  # MNISTを2値化
    train_data = datasets.MNIST(root="./", train=True, transform=transform, download=download_MNIST)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = datasets.MNIST(root="./", train=False, download=download_MNIST, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 学習
    rbm = RBM(n_visible_nodes=n_visible_nodes, m_hidden_nodes=m_hidden_nodes, epoch=epoch, lr=lr)
    RMSE_list = rbm.fit(train_loader=train_loader)

    # 訓練データでの再構成
    batch, _ = next(iter(test_loader))
    batch_reshaped = batch[:result_show_num].view(-1, n_visible_nodes)
    v_recon = rbm(batch_reshaped)

    # RMSEのエポックごとの推移表示
    plt.subplots()
    plt.plot(range(1, len(RMSE_list) + 1), RMSE_list, marker="o")
    plt.title("Changes in RMSE by epoch")
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.grid(True)

    # 再構成結果の表示
    fig, axes = plt.subplots(2, result_show_num, figsize=(12, 4))
    fig.suptitle("Test Raw Data (Top), Reconstruction Results (Bottom)")
    for i in range(result_show_num):
        axes[0, i].imshow(batch_reshaped[i].view(28, 28), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(v_recon[i].detach().numpy().reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()
