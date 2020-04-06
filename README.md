# PyTorch_Geometric_sample
自分用にPyTorch Geometricのサンプルを作っていきます。\
基本的には1つのpythonファイルで完結するようにします。

## install
1. pytorch等、各種基本ライブラリをインストール
2. pytorch geometricをインストール：https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

## check.py
pytorch geometricが正常にインストールできているか確認するためのファイル\
coraデータセットのダウンロード及びバージョンが表示されたら成功。
## mnist_sample.py
参考サイト:https://qiita.com/DNA1980/items/8c8258c9a566ea9ea5fc \
mnistデータセットを[ここ](http://yann.lecun.com/exdb/mnist/)からダウンロード\
ダウンロードしたデータを、``./Datasets/MNIST/``に保存する。\
輝度値を元にグラフ構造に変換したmnistデータにグラフ畳み込みを行って分類を行う。\
デフォルトは150エポック
