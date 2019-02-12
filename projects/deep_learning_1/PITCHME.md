# 目次

- パーセプトロン
- ニューラルネットワーク
- ネットワークの学習
- 誤差逆伝播法
- 学習に関するテクニック
- 畳み込みニューラルネットワーク
- ディープラーニング

---

# 1. パーセプトロン

+++

## パーセプトロンとは

Blah x3

+++

## パーセプトロンの動作原理

入力信号$x_1, x_2$, 出力信号$y$, 重み$w_1, w_2$について\
ニューロンの発火閾値$\theta$

`\[
y = 
\begin{cases}
    0 \ \left( w_1x_1 + w_2x_2 \leq \theta \right) \\
    1 \ \left( w_1x_1 + w_2x_2 >    \theta \right)
\end{cases}
\]`

+++

### 重みについて

> 重みは、電流で言うところの「抵抗」に相当します。抵抗は電流の流れにくさを 決めるパラメータであり、抵抗が低ければ低いほど大きな電流が流れます。一 方、パーセプトロンの重みは、その値が大きければ大きいほど、大きな信号が 流れることを意味します。抵抗も重みも信号の流れにくさ(流れやすさ)をコ ントロールするという点では同じ働きをします。
(本書き写し)

+++

## AND ゲート

|$x_1$|$x_2$|$y$|
|:----|:----|--:|
|  0  |  0  | 0 |
|  1  |  0  | 0 |
|  0  |  1  | 0 |
|  1  |  1  | 1 |

これをパーセプトロンで表す<br> 
-> 真理値表を満たすように($w_1$,$w_2$,$\theta$)を決める <br>
-> e.g.) (0.5, 0.5, 0.7), (0.5, 0.5, 0.8)

+++

## NAND

|$x_1$|$x_2$|$y$|
|:----|:----|--:|
|  0  |  0  | 1 |
|  1  |  0  | 1 |
|  0  |  1  | 1 |
|  1  |  1  | 0 |

ANDの符号変換だけで良い<br>
-> e.g.) (-0.5, -0.5, -0.7)

---

## 1.2 パーセプトロンの実装

``` python
def AND(x1, x2, w1=1, w2=1, theta=1):
    theta = 1, 1, 1
    y = x1*w1 + x2*w2
    return int(y<=theta)
```
+++

### 式変換（閾値からバイアスへ）

$\theta \rightarrow -b$して移項

`\[
y = 
\begin{cases}
    0 \ \left( b + w_1x_1 + w_2x_2 \leq 0 \right) \\
    1 \ \left( b + w_1x_1 + w_2x_2 >    0 \right)
\end{cases}
\]`

+++

### コード

``` python
def AND(x1, x2, w1=1, w2=1, b=-1):
    
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    y = np.sum(w*x) + b
    #閾値を超えたら(バイアスより入力の重み付き和が大きければ1を返す)
    return int(y>0)

```
+++

### NAND
``` python
# 重みとバイアスのみ異なる！
def NAND(x1, x2, w1=-1, w2=-1, b=1):
    
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    y = np.sum(w*x) + b
    return int(y>0)
    
def OR(x1, x2, w1=0.5, w2=0.5, b=-0.2):
    
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    y = np.sum(w*x) + b
    return int(y>0)
```
下位関数作る？

+++

### ゲート関数

``` python
def _gate_func(x1, x2, w1, w2, b):
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    y = np.sum(w*x) + b
    #閾値を超えたら(バイアスより入力の重み付き和が大きければ1を返す)
    return int(y>0)
    
def AND(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.7
    return _gate_func(x1, x2, w1, w2, b)

def NAND(x1, x2):
    w1, w2, b = -0.5, -0.5, 0.7
    return _gate_func(x1, x2, w1, w2, b)
    
def OR(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.2
    return _gate_func(x1, x2, w1, w2, b)  
```

+++ 

### XOR: パーセプトロンの限界

入力のどちらかのみが1のときに出力1

|$x_1$|$x_2$|$y$|
|:----|:----|--:|
|  0  |  0  | 0 |
|  1  |  0  | 1 |
|  0  |  1  | 1 |
|  1  |  1  | 0 |

非線形は判別不可能

---

## 多層パーセプトロン

回路の組み合わせにより実現可能<br>
($NAND$, $OR$, $AND$)

|$x_1$|$x_2$|$s_1$|$s_2$|$y$|
|:----|:----|:---:|:---:|--:|
|  0  |  0  |  0  |  0  | 0 |
|  1  |  0  |  1  |  1  | 1 |
|  0  |  1  |  0  |  1  | 1 |
|  1  |  1  |  0  |  1  | 0 |

+++

``` python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
```

+++

NANDだけでコンピューターを作ることができる<br>
理論上2層のパーセプトロンであればコンピューターを作る


<!-- 
p.39
3章
ニューラルネットワーク
-->

---

### 2．ニューラルネットワーク

+++

#### パーセプトロン Pros＆Cons

- Pros
    - 複雑な関数を表現できる可能性がある
- Cons
    - 適切な重みを決める作業は今の所人力で行われている

→ NNはこの問題を解決するためにある<br>
= 適切な重みパラメータをデータから自動で学習できる

---

#### 3.1 パーセプトロンとNNの相違点

- ニューロンの繋がり方は同じ
- NNは活性化関数を用いる
+++

- パーセプトロンの式
`\[
y = 
\begin{cases}
    0 \ \left( b + w_1x_1 + w_2x_2 \leq 0 \right) \\
    1 \ \left( b + w_1x_1 + w_2x_2 >    0 \right)
\end{cases}
\]`

- ニューラルネットワークの式

`\[y = h(b + w_1x_1 + w_2x_2)\]`
`\[
h(x) = 
\begin{cases}
    0 \ \left( x \leq 0 \right) \\
    1 \ \left( x >    0 \right)
\end{cases}
\]`

このときの`h(x)`を **活性化関数** という

+++

#### 活性化関数の意味合い

どのような入力の総和が活性化するかを決定する．

1. 重み付きの入力信号の総和を計算
2. その和が活性化関数によって変換される

--- 

#### 3.2 活性化関数

+++

#### シグモイド関数

`\[h(x) = \frac{1}{1+exp(-x)} \]`

?-> ステップ関数のような役割で，出力を連続値にしたい場合に有効？

+++ 

#### ステップ関数の実装

``` python
def step_function(x):
    """
    入力は実数(float)に限られ，配列を使うことができない
    """
    if x<=0:
        return 0
    return 1
```

+++ 

#### 配列に対して処理可能にする

``` python

def step_function(x):
    """
    NumPyの配列を想定
    """
    y = x > 0
    return y.astype(np.int)
```

+++

#### シグモイド関数

``` python
def sigmoid(x):
    return 1/(1 + np.exp(-x))
```

- 入力値が大きければ，大きな出力
- 入力信号の値が大きくても，出力が[0, 1]に収まる

+++

#### 非線形関数を使う意味

線形関数を用いるとNNで層を深くすることの意味がなくなる

- 例：
$h(x) = cx$を活性化関数とすると$y(x) = h(h(h(x))) = c^3x$とするのは$c^3=a$と代入したときの$y(x)=ax$に相当し，3回計算する必要がない．

+++

#### ReLU関数

ReLU(Rectified Linear Unit)という関数が主に用いられる．
`基本は線形関数`
`\[
y(x) = 
\begin{cases}
    x \ \left( x >    0 \right) \\
    0 \ \left( x \leq 0 \right)
\end{cases}
\]`

```python
def relu(x):
    return np.maximum(0, x)
```

+++

#### 多次元配列でNNの実装を効率的に

入力層$X = x_1, x_2$, 出力層$y_1, y_2, y_3$に対して<br>
重み行列$W = \left(
    \begin{array}{cc}
      w_11 & w_12 & w_13 \\
      w_21 & w_22 & w_23 
    \end{array}
  \right)

``` python
X = np.array([1, 2])
W = np.array([[1, 3, 4], [2, 4, 6]])

Y = np.dot(X, W)
```

+++ 

#### 3層NNの実装

### $w^{(l)}_{ba}$

$l$層目の重み（前層$b$番目のニューロンから次層$a$番目のニューロン）

次層における出力$a_1^{(1)}$は

$a_1^{(1)}　= w^{(1)}_{11}x_1 + w^{(1)}_{12}x_2 + b_1^{(1)}$

+++

#### 入力(0)層から隠れ第一(1)層, 1層から隠れ第二(2)層まで

``` python
A1 = np.dot(X, W1) + B1
# シグモイド関数に突っ込む
Z1 = sigmoid(A1)

# 中略

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
```

+++

#### 隠れ第二(2)層から出力(3)層まで

活性化関数が異なる

``` python
def identity_function(x):
    #恒等関数，入力をそのまま返す
    #`\sigma()`
    return x

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
```

+++ 実装のまとめ

``` python
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
```

+++ 

#### 出力層に使う関数について

一般に
- 恒等関数: 回帰問題
- ソフトマックス関数: 分類問題

+++

#### ソフトマックス関数

$y_k = \frac{exp(a_k){\Sigma_{i=1}^n exp(a_i)}$

``` py
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```

理論上は正しいが，実装上は値が大きくなりすぎてオーバーフローする

+++ 

#### 改善策
`\[
y_k = \frac{exp(a_k){\Sigma_{i=1}^n exp(a_i)} \\
    = \frac{C exp(a_k){C \Sigma_{i=1}^n exp(a_i)} \\
    = \frac{exp(a_k + logC){\Sigma_{i=1}^n exp(a_i+logC)}
\]`
ここで$logC = C'$とする．$C'$には入力の中での最大値を使うことが一般的

``` py
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```

+++

#### ソフトマックス関数の特徴

- 出力は0から1.0までの実数
- 出力の総和は1
    - よって，出力結果を確率として解釈可能
- ? 単純な割合ではいけないのか?（負の値に対する処理のため？）
  

--- 

## 実データ例: MNIST

+++

### データの読み込み

``` py
import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするため
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# 28x28=784画素に関するデータが6万文字分ある
# flatten=Trueのため，画素は1次元で表現される
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000,)
print(x_test.shape) # (10000, 784)
print(t_test.shape) # (10000,)

```

+++

### データの迅速な取り出しのために

1回目はネット接続で読み込むため，時間がかかるが2回目は`pickle`ファイルを読み込むだけなため，処理が早く終了する．

``` py
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network
```

+++

予測 = NNを回してsoftmaxで最も高い値を持つものを最も出る確率が高いとみなす

```py
x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # 最も確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```

大体94%くらい

---

### 精度向上に向けて

+++ 

#### バッチ処理

##### 入力データと重みパラメータの形状に注目する
データをまとまりで処理する（複数画像を同時に処理する）

+++ 

``` py
x, t = get_data()
network = init_network()
batch_size = 100 # バッチの数
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    #batch_size分のデータを入力とする
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```


---

## 3.ニューラルネットワークの学習

トレーニングデータから最適な重みパラメータの値を自動で獲得

- 損失関数を指標として学習する
- 勾配法を用いる
- 数千，数万のパラメータを手動で設定することは不可能

+++

### Data-driven

手書き数字を判別するアルゴリズムを考えるのは難しい．<br>
→データを利用する!

- 特徴量を使う(データをベクトルに変換)
    - 画像解析では...
        - SIFT
        - SURF
        - HOG
- SVMやKNNで学習

- Deep Learningは画像をそのまんま全て学習！

---

### モデル評価

汎化能力：訓練データに含まれないデータに対しての能力

損失関数(loss function)：モデルの性能評価をする

+++

#### 2乗和誤差(Mean Squared Error)

$E=\frac 1 2 \Sigma_k{(y_k-t_k)^2}$

``` python
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
```

+++ 

#### 交差エントロピー誤差(Cross Entropy Error)

$E=-\Sigma_k{t_k log y_k}$

正解ラベルに対応する出力が小さければ値が大きくなる

``` python
def cross_entropy_error(y, t):
    #np.log(0)->-infを防ぐため
    delta = 1e-7
    return -np.sum(t*np.log(y + delta)
```

---

### ミニバッチ学習

+++

訓練データを使った学習：訓練データに対する損失関数をできるだけ小さくするパラメータを探し出す．<br>
→損失関数は全ての訓練データを対象として求める必要がある．

交差エントロピー誤差の場合<br>
$E=-\frac{1}{N}\Sigma_n\Sigma_k{t_{nk} log y_{nk}}$

+++

訓練データが例えば6万個あったとき，全てを対象にするのは時間がかかる．<br>
一部のデータを全体の近似として利用する．この一部のデータを**ミニバッチ**という．<br>
大量のデータから例えば100枚を**無作為に**取り出す必要がある．

+++

全データをインポート

``` python
import sys, os
sys.path.append(os.pardir) #dataset.mnist読み込みのため
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
load_mnist(normalize=True, one_hot_label=True)
```

+++

ランダムに100個インポート

```
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
```
+++

交差エントロピー誤差の実装
```
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    #(one-hot)
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
    #2や7など正解をそのまま返す場合
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```

+++

「認識精度」ではなく「損失関数」で評価する理由

>認識精度を指標にすると，パラメータの微分がほとんどの場所で0になってしまう

100枚の写真を認識するとき，認識精度は$33%, 34%, ...$と不連続な変化をする．

? 認識精度関数のような形で連続的変化をする関数を作ればよいのでは？

---

### 数値微分

勾配法：勾配を利用する

+++

定義

`\[
\frac{df(x)}{dx} = lim_{h\to0}{\frac{f(x+h)-f(x)}{h}}
\]`

``` python
#Bad Implementation
def numerical_diff(f,x):
    #値が小さすぎて丸め誤差が生じる
    h = 10e-50
    
    #hをあまり小さくできないため誤差が大きい
    return (f(x+h)-f(x))/h
```

+++

改善

``` python 

def numerical_diff(f, x):
    h = 1e-4
    #中心差分を取る
    return (f(x+h)-f(x-h) / (2*h)
```

+++

例

$ y = 0.01x^2 + 0.1x$

``` python
def f1(x):
    return 0.01*x**2 + 0.1*x
```

``` python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0, 20.0, 0.1)
y = f1(x)
plt.plot(x, y)
plt.show()
```

+++

2変数の二乗和の偏微分

$f(x_0,x_1) = x_0^2 + x_1^2$

``` python
def f2(x):
return x[0]**2 + x[1]**2
#return np.sum(x**2)
```

``` python

# 変数が1つだけの関数を定義して，その関数について微分を求める
# 入力に対して下のような関数を定義するのは少し面倒
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1
>>> numerical_diff(function_tmp2, 4.0)
7.999
```

+++

勾配の計算

``` python
def numerical_gradient(f, x):
    h = 1e-4
    # xと同じ形状の配列
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        #f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        #f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad
```

+++

### 勾配法
各地点においてその関数を減少させる方向を示す
+++

`\[
x_0 = x_0 - \eta \frac{\delta f}{\delta x_0} \
x_1 = x_1 - \eta \frac{\delta f}{\delta x_1}
\]`

$\eta$: 学習率(learning rate)

個の更新ステップを繰り返す行う

+++

Pythonによる実装

``` python
def gradient_dscent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x
```

+++ 

SimpleNetクラスの実装

``` python
class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)#ガウス分布で初期化
    
    def predict(self, x):
        return np.dot(x, self.W)
        
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss        
```

``` python

net = SimpleNet()
print(net.W)
p = net.predict([0.6, 0.9])

np.argmax(p) 最大値のインデックス

t = np.array([0, 0, 1])#正解
net.loss(x, t)
```

---

### 学習アルゴリズム

確率的勾配降下法(SGD: Stochastic Gradient Descent)

+++


