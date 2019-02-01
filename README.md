# i-i-i-work-out

feat. LMFAO

## GitPitch
(url: https://gitpitch.com/docs/git/branch-many-slideshows/)

template: https://gitpitch.com/$username/$repo/$branch?p=$rep


### ディープラーニング
- https://gitpitch.com/gshirato/i-i-i-work-out?p=projects/deep_learning_1
- https://gitpitch.com/gshirato/i-i-i-work-out/perceptron?p=projects/deep_learning_1
### 統計
- https://gitpitch.com/gshirato/i-i-i-work-out?p=projects/statistics

### Use image

![alt](assets/image_name.png)

---?assets/bg.png

* 箇条書き1
* 箇条書き2

# Miscellaneous

## 右クリックでからのファイル作成

参考: https://qiita.com/sugasaki/items/d52c33ea8ad6b74c052e

1. `Automator` -> `新規書類` -> `サービス(歯車マーク)`

2. `ワークフローが受け取る現在の項目: ファイルまたはフォルダ`

3. 左窓`ユーティリティ` -> `AppleScriptを実行`

4. 下のコードをコピペ，ハンマーアイコン

```
on run {input, parameters}
  tell application "Finder"
    set currentPath to insertion location as text
    set x to POSIX path of currentPath
  end tell
  return x
end run
```

5. 左窓`ライブラリ` -> `ユーティリティ` -> `変数の値を設定`を右側の空白エリアにドラッグ＆ドロップ

5.2 新規変数 -> `カレントフォルダ`と命名（例）

6. 左窓`ライブラリ`->`テキスト`->`新規テキストファイル`で保存場所を`カレントフォルダ` 
