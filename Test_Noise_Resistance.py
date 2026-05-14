import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

df = pd.read_csv('dfi_training_data.csv')
X = df.drop('Risk_Level', axis=1)
y = df['Risk_Level']

chemical_cols = ['Furanocoumarins', 'Tyramine', 'Vitamin_K', 'Potassium', 'Bleeding_Enhancer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1), 
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)

noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
results = []

print("\n開始測試")

for noise in noise_levels:
    X_test_noisy = X_test.copy()
    
    if noise > 0:
        noise_matrix = np.random.normal(0, noise, size=(X_test.shape[0], len(chemical_cols)))
        X_test_noisy[chemical_cols] += noise_matrix
        X_test_noisy[chemical_cols] = np.clip(X_test_noisy[chemical_cols], 0, 1.5)
    
    for name, model in models.items():
        start_time = time.perf_counter() 
        y_pred = model.predict(X_test_noisy)
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            "Noise Level": noise,
            "Model": name,
            "Accuracy": acc,
            "Inference Time (sec)": elapsed_time 
        })
        
    print(f"雜訊強度 {noise:.2f} ")

results_df = pd.DataFrame(results)

pivot_acc = results_df.pivot(index='Noise Level', columns='Model', values='Accuracy')
print("\n各模型於不同視覺雜訊強度下之準確率變化表")
print(pivot_acc.to_markdown(floatfmt=".4f"))

time_summary = results_df.groupby('Model')['Inference Time (sec)'].mean().reset_index()
print("\n各模型平均推論耗時對比")
print(time_summary.to_markdown(index=False, floatfmt=".6f"))

print("\n繪製抗干擾衰減曲線圖")

plt.figure(figsize=(10, 6))

palette = {
    "SVM": "#e74c3c",          
    "Random Forest": "#3498db",
    "XGBoost": "#2ecc71",      
    "LightGBM": "#f39c12"      
}

sns.lineplot(
    data=results_df, 
    x='Noise Level', 
    y='Accuracy', 
    hue='Model', 
    palette=palette,  
    marker='o',       
    linewidth=2.5,    
    markersize=8      
)

plt.title('決策模型對視覺雜訊之抗干擾衰減曲線', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('視覺雜訊強度 (Noise Level)', fontsize=14)
plt.ylabel('預測準確率 (Accuracy)', fontsize=14)


plt.xticks(noise_levels)
plt.yticks(np.arange(0.84, 1.01, 0.02)) 
plt.ylim(0.83, 1.01) # 給 Y 軸留一點點上下邊界
plt.grid(True, linestyle='--', alpha=0.7)

plt.legend(title='演算法模型', fontsize=12, title_fontsize=12, loc='lower left')

plt.tight_layout()
plt.savefig('final_robustness_degradation_curve_with_lgbm.png', dpi=300) 
plt.show()

print("圖表存檔")