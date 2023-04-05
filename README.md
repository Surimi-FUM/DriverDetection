# DriverDetection
Deep Learningで車のペダル踏み間違い事故を防止するプロジェクトで作成した  

![predict11](https://user-images.githubusercontent.com/116449282/229400077-9d2af224-7ed6-402b-9e3c-127ebfc17702.jpg)  

![predict10](https://user-images.githubusercontent.com/116449282/229400079-a11dd10d-7888-4bb8-931d-22496a88831f.jpg)  

当初はCNNによる分類を行ったが、ペダルがある領域が暗すぎるためCNNでは足元を検出することができなかった。  
対策として、手動でセグメンテーションを施した画像を使用したら精度が向上した。  
その結果から、セグメンテーションモデルが有効と考え、今回のUNetの実装を行った。  
教師データは、用意した独自データセットではデータの多様性が乏しかったため、公開されているPascal VOC 2012を使用することにした。  

## 結果  


## 開発環境
PC：Windows 10 Education、Intel Core i7-7700K  
メモリ：32.0GB  
GPU：NVIDIA GeForce GTX 1080  

### テスト環境
開発環境と同じ  

## 開発期間・人数  
令和3年(2021)/05 ~ 令和4年(2022)/02、1人  
令和4年(2022)/05 ~ 令和5年(2023)/02、1人  
