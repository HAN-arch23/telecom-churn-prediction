import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve)
from xgboost import XGBClassifier
import shap

SEED = 42
np.random.seed(SEED)

# ── SECTION 1: PROBLEM DEFINITION ──────────────────────────
print("SECTION 1: PROBLEM DEFINITION")
print("""
Problem: Predict whether a telecom customer will churn (Yes/No).
Goal: Build a binary classification model to enable proactive retention.
Task Type: Binary Classification
Impact: Reduces revenue loss from customer attrition.
""")

# ── SECTION 2: DATA ACQUISITION ────────────────────────────
print("SECTION 2: DATA ACQUISITION")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"Shape: {df.shape} | Target: Churn | Source: IBM/Kaggle")
print(df.head(3))

# ── SECTION 3: DATA CLEANING ───────────────────────────────
print("\nSECTION 3: DATA CLEANING")
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print(f"Missing values: {df.isnull().sum().sum()} | Duplicates: {df.duplicated().sum()}")
print(f"Churn distribution:\n{df['Churn'].value_counts()}")

# ── SECTION 4: EDA ─────────────────────────────────────────
print("\nSECTION 4: EDA")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('EDA – Telecom Churn', fontsize=16, fontweight='bold')

# Churn distribution
churn_counts = df['Churn'].value_counts()
axes[0,0].bar(['No Churn','Churn'], churn_counts, color=['steelblue','tomato'])
axes[0,0].set_title('Churn Distribution')

# Tenure by churn
df.boxplot(column='tenure', by='Churn', ax=axes[0,1])
plt.sca(axes[0,1]); plt.title('Tenure by Churn')

# Monthly charges
axes[0,2].hist(df[df['Churn']==0]['MonthlyCharges'], bins=30, alpha=0.6, label='No Churn', color='steelblue')
axes[0,2].hist(df[df['Churn']==1]['MonthlyCharges'], bins=30, alpha=0.6, label='Churn', color='tomato')
axes[0,2].set_title('Monthly Charges'); axes[0,2].legend()

# Contract type churn rate
contract_churn = df.groupby('Contract')['Churn'].mean()
axes[1,0].bar(contract_churn.index, contract_churn.values, color=['steelblue','orange','tomato'])
axes[1,0].set_title('Churn Rate by Contract')

# Internet service churn rate
internet_churn = df.groupby('InternetService')['Churn'].mean()
axes[1,1].bar(internet_churn.index, internet_churn.values, color=['steelblue','orange','tomato'])
axes[1,1].set_title('Churn Rate by Internet Service')

# Correlation heatmap
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1,2])
axes[1,2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# ── SECTION 5: FEATURE ENGINEERING ────────────────────────
print("\nSECTION 5: FEATURE ENGINEERING")
df['ChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)

service_cols = ['PhoneService','MultipleLines','InternetService','OnlineSecurity',
                'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
df['NumServices'] = df.apply(
    lambda r: sum(1 for c in service_cols if r[c] not in ['No','No phone service','No internet service']),
    axis=1)

binary_cols = ['gender','Partner','Dependents','PhoneService','PaperlessBilling']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

multi_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
              'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
              'Contract','PaymentMethod']
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

scaler = StandardScaler()
num_cols = ['tenure','MonthlyCharges','TotalCharges','ChargesPerMonth','NumServices']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")

# ── SECTION 6: MODEL BUILDING ──────────────────────────────
print("\nSECTION 6: MODEL BUILDING")
models = {
    'Logistic Regression': LogisticRegression(random_state=SEED, max_iter=1000),
    'Decision Tree'      : DecisionTreeClassifier(random_state=SEED, max_depth=5),
    'Random Forest'      : RandomForestClassifier(random_state=SEED, n_estimators=100),
    'XGBoost'            : XGBClassifier(random_state=SEED, eval_metric='logloss', n_estimators=100)
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
    print(f"  {name:<25} | CV F1: {cv:.4f}")

# Tune XGBoost
param_grid = {'max_depth':[3,5], 'learning_rate':[0.05,0.1], 'n_estimators':[100,200]}
xgb_grid = GridSearchCV(XGBClassifier(random_state=SEED, eval_metric='logloss'),
                         param_grid, cv=3, scoring='f1', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
trained_models['XGBoost (Tuned)'] = xgb_grid.best_estimator_
print(f"  Best XGBoost Params: {xgb_grid.best_params_}")

# ── SECTION 7: MODEL EVALUATION ────────────────────────────
print("\nSECTION 7: MODEL EVALUATION")
results = []
for name, model in trained_models.items():
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    results.append({'Model': name,
                    'Accuracy' : round(accuracy_score(y_test, y_pred), 4),
                    'Precision': round(precision_score(y_test, y_pred), 4),
                    'Recall'   : round(recall_score(y_test, y_pred), 4),
                    'F1-Score' : round(f1_score(y_test, y_pred), 4),
                    'ROC-AUC'  : round(roc_auc_score(y_test, y_proba), 4)})

results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
print(results_df.to_string(index=False))

best_model  = trained_models['XGBoost (Tuned)']
y_pred_best = best_model.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_best),
                       display_labels=['No Churn','Churn']).plot(ax=axes[0], cmap='Blues')
axes[0].set_title('Confusion Matrix – XGBoost (Tuned)')

for name, model in trained_models.items():
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    axes[1].plot(fpr, tpr, label=f'{name} ({roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.3f})')
axes[1].plot([0,1],[0,1],'k--'); axes[1].set_title('ROC Curves')
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR'); axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()

# ── SECTION 8: INSIGHTS ────────────────────────────────────
print("\nSECTION 8: INSIGHTS")
feat_imp = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)
feat_imp.plot(kind='barh', color='steelblue', figsize=(10,6), title='Top 15 Features')
plt.gca().invert_yaxis(); plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight'); plt.show()

explainer = shap.TreeExplainer(best_model)
shap_vals = explainer.shap_values(X_test)
shap.summary_plot(shap_vals, X_test, plot_type='bar', max_display=15, show=False)
plt.tight_layout(); plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight'); plt.show()

print("""
Insights:
  1. Tenure is the #1 predictor — short-tenure customers churn most.
  2. Month-to-month contracts = highest churn risk.
  3. High monthly charges strongly predict churn.
  4. Fiber optic users churn more than DSL users.
  5. Customers without security/support services churn more.

Recommendations:
  - Offer loyalty discounts in first 12 months.
  - Incentivize annual/2-year contract sign-ups.
  - Bundle security + support for fiber optic customers.
  - Proactively contact high monthly charge + new customers.
""")

print("DONE! Output files: eda_plots.png, model_evaluation.png, feature_importance.png, shap_summary.png")