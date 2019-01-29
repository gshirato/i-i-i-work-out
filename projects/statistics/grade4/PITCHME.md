# 統計検定４級
## 資料の活用



---

### 統計分析

|No.|Name|Sex|Sports|Commute Time|
|:--|---:|--:|-----:|-----------:|
| 1 |Lucy| F |Soccer|          14|
| 2 |Mack| M |Hockey|          20|

行：一件分のデータ<br>
列：データの項目

+++

### この学校はどのスポーツが人気？

150人のうち3分の1がサッカーが好き<br>
*ばらついている*=好きなスポーツはそれぞれちがう<br>
分布 = ばらつきの様子

<canvas data-chart="bar">
<!--
{
 "data": {
  "labels": ["Soccer"," Rugby"," Baseball"," Tennis"," Hockey"],
  "datasets": [
   {
    "data":[50, 30, 25, 35, 10],
    "label":"Sports","backgroundColor":"rgba(20,220,220,.8)"
   }
  ]
 }, 
 "options": { "responsive": "true",
               "scales": 
                {
                "xAxes": [{
                    "stacked": true
                }],
                "yAxes": [{
                    "stacked": true
                }]
            }
    }
 }
-->
</canvas>


+++

>平均x人がサッカー好きで，分散がいくつで～～・

データの特徴を知る→記述統計学

>他の学校でもw人がサッカー好きだろう

知らないデータの予測→推測統計学

---

## グラフ

+++

### ポイント

1. データの種類を判断する
2. データの種類・内容にあったグラフを選ぶ

+++

### 棒グラフ

他のデータとの比較に便利

<canvas data-chart="bar">
<!--
{
 "data": {
  "labels": ["ボスニア"," オランダ"," 日本"," ナイジェリア"," ベトナム"],
  "datasets": [
   {
    "data":[183.9, 183.8, 170.7, 163.8, 162.1],
    "label":"Sports","backgroundColor":"rgba(20,220,220,.8)"
   }
  ]
 }, 
 "options": { "responsive": "true",
               "scales": 
                {
                "xAxes": [{
                    "ticks": {
                        "beginAtZero":"true"
                    } 
                }],
                "yAxes": [{
                    "ticks": {
                        "beginAtZero":"true"
                    } 
                }]
            }
    }
 }
-->
</canvas>

+++

### 折れ線グラフ

時間の変化を記述するのに便利

<canvas data-chart="line">
<!--
{
 "data": {
  "labels": ["2012", "2013", "2014", "2015", "2016"],
  "datasets": [
   {
    "data":[6.2, 5.2, 4.9, 4.4, 4.9],
    "label":"Japan's GDP","backgroundColor":"rgba(20,220,220,.8)"
   }
  ]
 }, 
 "options": { "responsive": "true",
               "scales": 
                {
                "xAxes": [{
                      "ticks": {
                        "beginAtZero":"true"
                       } 
                }],
                "yAxes": [{
                    "ticks": {
                       "beginAtZero":"true"
                    } 
                }]
            }
    }
 }
-->
</canvas>

+++

度数と相対度数

+++

幹葉図

<left>
 
 3|1
 2|222
 1|314155
 0|131
</left>
