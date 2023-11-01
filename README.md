# CoLaboratory 訓練神經網路

本文旨在展示如何使用CoLaboratory 訓練神經網路。我們將展示一個在威斯康辛乳癌資料集上訓練神經網路的範例，資料集可在UCI Machine Learning Repository（http://archive.ics.uci.edu/ml/datasets）取得。本文的範例相對比較簡單。

本文所使用的CoLaboratory notebook 連結：https://colab.research.google.com/notebook#fileId=1aQGl_sH4TVehK8PDBRspwI4pD16xIR0r

# 深度學習

深度學習是一種機器學習技術，它所使用的計算技術一定程度上模仿了生物神經元的運作。各層中的神經元網路不斷將資訊從輸入傳輸到輸出，直到其權重調整到可以產生反映特徵和目標之間底層關係的演算法。

想更了解神經網絡，推薦閱讀這篇論文《Artificial Neural Networks for Beginners》（https://arxiv.org/pdf/cs/0308031.pdf）。

程式碼
問題：研究者取得乳房腫塊的細針穿刺（FNA），然後產生數位影像。此資料集包含描述影像中細胞核特徵的實例。每個實例包括診斷結果：M（惡性）或B（良性）。我們的任務是在該數據上訓練神經網路根據上述特徵診斷乳癌。

開啟CoLaboratory，出現一個新的untitled.ipynb 檔案供你使用。

谷歌允許使用其伺服器上的一台linux 虛擬機，這樣你可以存取終端為專案安裝特定套件。如果你只在程式碼單元中輸入!ls 指令（記得指令前加!），那麼你的虛擬機器中會出現一個datalab 資料夾。
![image](https://github.com/dtanlley/report1/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202023-11-01%20122739.png)
