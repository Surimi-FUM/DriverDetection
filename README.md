# DriverDetection
Deep Learningで車のペダル踏み間違い事故を防止するプロジェクトで作成した  

![predict12](https://user-images.githubusercontent.com/116449282/229969539-f1f97d02-dc72-46ce-b2e5-09abd3bb2d19.jpg)  

![predict1](https://user-images.githubusercontent.com/116449282/229969756-98147c59-160b-46d9-9c6b-e3dc3e09d374.jpg)  

当初はCNNによる分類を行ったが、ペダルがある領域が暗すぎるためCNNでは足元を検出することができなかった。  
対策として、手動でセグメンテーションを施した画像を使用したら精度が向上した。  
その結果から、セグメンテーションモデルが有効と考え、今回のUNetの実装を行った。  
教師データは、用意した独自データセットではデータの多様性が乏しかったため、公開されているPascal VOC 2012を使用することにした。  

## 結果  
推論モデルとして、2015年に提案された標準モデル、標準モデルにバッチ正規化層を追加したモデル、損失関数をDice係数lossに変更したモデル、二つを組み合わせたモデルの4種類を構築した。  
それぞれunet、unetBN、unetDL、unetBNDLに区分する。  
検証データ178枚に対するMeanIoUを計算した結果は(unet, unetDL, unetBN, unetBNDL) = (0.4179, 0.4154, 0.5740, 0.6087)となった。  
この結果からunetBNDLが一番精度のよいモデルであると推測される。  
unetBNDLでは、他モデルと比べてセグメンテーション領域が広がり改善がみられた。  
しかし、セグメンテーション領域は途切れ途切れな状態であり、ハンドルを持つ腕部分は大雑把に過剰反応していた。  
出力されたマスク画像から人体像を認識することはできない精度ではあるものの、今までの進捗の中でとらえられていなかった足元に対して反応している結果が得られた。  

## 開発環境
PC：Windows 10 Education、Intel Core i7-7700K  
メモリ：32.0GB  
GPU：NVIDIA GeForce GTX 1080  

### テスト環境
開発環境と同じ  

## 開発期間・人数  
令和3年(2021)/05 ~ 令和4年(2022)/02、1人  
令和4年(2022)/05 ~ 令和5年(2023)/02、1人  
