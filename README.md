# 基於雙重可解釋性人工智慧之藥食交互作用預測系統：結合影像辨識與非線性決策模型
### A Drug-Food Interaction Prediction System Based on Dual Explainable AI: Integrating Image Recognition and Non-linear Decision Models

![](https://img.shields.io/badge/Python-3.11.9-blue.svg)
![](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)
![](https://img.shields.io/badge/Classifier-XGBoost-green.svg)
![](https://img.shields.io/badge/XAI-Grad--CAM%20%2F%20SHAP-orange.svg)

## 📌 專案簡介
本專案為國立彰化師範大學資訊管理學系碩士論文之完整實驗程式碼。

現行飲食紀錄系統多忽略**藥食交互作用（Drug-Food Interaction, DFI）**之潛在風險，且缺乏台灣在地飲食數據與決策透明度。為解決此臨床痛點，本研究開發出一套**雙重可解釋性人工智慧（Dual XAI）預警系統**。系統針對 17 類高風險台灣常見食材建置影像資料集，並提出**「跨模態特徵映射機制」**，將前端影像辨識的機率分佈向量，轉譯為後端臨床可計算的化學成分特徵值。最後結合非線性決策模型進行多分類風險等級判定，並提供「影像端 + 決策端」的雙重可解釋性分析，有效輔助病患與醫療人員進行用藥安全把關。

---

## ⚡ 核心技術
1. **前端在地化影像辨識 (Localized Image Recognition)**
   本研究採用 ResNet50 卷積神經網路架構，針對台灣日常飲食（如香腸、臘肉、花椰菜、木瓜等 17 類具潛在交互作用風險之食材）自行建置並擴充專屬影像資料集。模型在複雜的日常飲食背景下整體辨識準確率達 78.12%，對於具備高鑑別度之食材（如白飯、蒜頭、啤酒）更具備完美的特徵捕捉能力，為系統提供穩定且可靠的視覺特徵來源。
   
2. **跨模態特徵映射機制 (Cross-Modal Feature Mapping)**
   有別於傳統系統僅依賴前端影像模型的最高機率（Top-1）分類輸出，本研究保留完整的機率分佈向量。當遭遇外觀相似的食材（如臘肉與香腸、木瓜與芒果）或影像模糊時，映射機制能將視覺不確定性平滑收斂為化學成分特徵值（如：呋喃香豆素、酪胺、維生素 K、鉀離子、出血增強物質），增加的容錯率與抗干擾能力。
   
3. **非線性決策模型**
   後端比較了 SVM、Random Forest、LightGBM 與 XGBoost 等多種演算法，選定具備最佳泛化能力與推論速度的 XGBoost 模型，後端決策模型能透過連續型化學特徵展現穩健的抗雜訊衰減效能。
   
4. **雙重可解釋性架構 (Dual Explainable AI)**
   * **視覺解釋端 (Grad-CAM)**：動態映射卷積層特徵圖，直觀呈現模型對食材色彩、紋理與形狀的空間注意力。
   * **決策解釋端 (SHAP)**：採用 TreeExplainer 拆解非線性決策路徑，透過瀑布圖（Waterfall Plot）明確量化各項化學特徵值與病患用藥紀錄的紅藍（正負向）推力貢獻度，決策邏輯完美契合真實臨床藥理機制（如 CYP3A4 酵素抑制、乳酪效應、高血鉀症及藥物拮抗作用）。

---

## 🗂️ 檔案結構
```📂 My_project
├── 📂 dataset               # 原始影像資料集 (未進行資料增強)
│   ├── 📂 train             # 訓練集 (內含 17 類在地食材子資料夾，如 Rice, Banana, Broccoli 等)
│   └── 📂 val               # 驗證集 (內含 17 類在地食材子資料夾)
├── 📂 dataset_augmented     # 擴充影像資料集 (經 Data Augmentation 幾何與色彩變換處理)
├── 📂 dataset_test          # 獨立測試集 (存放用於跨情境泛化驗證之盲測影像)
├── 📂 demo                  # 系統展示與 XAI 視覺化報告產出介面相關模組
├── 📄 WebCrawler.py         # 自動化網路爬蟲 (基於 icrawler 套件建置 17 類食材影像)
├── 📄 Train_ResNet50.py     # 前端視覺特徵擷取模型訓練程式
├── 📄 Demo_System.py        # 主要程式
├── 📄 Train_XGBoost.py      # 後端風險決策模型訓練與超參數調校程式
├── 📄 Data_Augmentation.py  # 資料增強程式
└── 📄 Test_Noise_Resistance.py   # 決策模型抗雜訊、抗干擾性與推論延時衰減測試驗證

## 執行環境與核心套件
* Python 3.11.9
* PyTorch 
* XGBoost / Scikit-learn / LightGBM 
* SHAP / Grad-CAM 
* OpenCV / PIL 
* Pandas / NumPy
