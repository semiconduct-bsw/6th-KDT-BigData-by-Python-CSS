# 1. 혼동 행렬(Confusion Matrix) 시각화
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=wine.target_names, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 2. 특성 중요도(Feature Importance) 시각화
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # 중요도 내림차순 정렬

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), np.array(wine.feature_names)[indices], rotation=45, ha='right')
plt.tight_layout()
plt.show()
