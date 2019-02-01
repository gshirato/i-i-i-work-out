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

##　3.ニューラルネットワークの学習

トレーニングデータから最適な重みパラメータの値を自動で獲得

- 損失関数を指標として学習する
- 勾配法を用いる