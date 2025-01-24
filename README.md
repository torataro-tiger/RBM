# RBM
制限ボルツマンマシン実行用のコードです。

## 実行方法
仮想環境を作成し`poetry install`でモジュールをインストールするか、`pip install -r requirements.txt`でモジュールをインストールしてください。

初めてコードを実行するときは以下のコマンドでMNISTをダウンロードしてから実行します。ダウンロードする際は65MB程度の空き容量を確保してください。
```
python RBM_run.py --download_MNIST
```

MNISTダウンロード後は以下のコマンドでも実行できます。
```
python RBM_run.py
```

またコマンドライン引数を指定して実行できます。バッチサイズや隠れ層の大きさなどを変更できます。
```
python RBM_run.py \
    --batch_size 64 \
    --result_show_num 10 \
    --download_MNIST \
    --m_hidden_nodes 64 \
    --epoch 5 \
    --lr 0.02
```

コマンドライン引数の内容は以下のコマンドで確認してください。
```
python RBM_run.py --help
```

## 実行結果例
各エポックごとのRMSE（正確にはエポック内で計算したRMSEの平均）<br/>
![](/images/fig1.png)

再構成結果（上：生テスト画像、下：再構成結果）<br/>
![](/images/fig2.png)
