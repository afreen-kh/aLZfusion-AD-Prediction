"""
Alzheimer's Prediction — aLZfusion ML Pipeline
Stacking ensemble, SHAP, and publication-quality figures.
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import itertools
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from mlxtend.classifier import StackingCVClassifier
import xgboost as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# STEP 1: Load and prepare data
# ============================================================
print("=" * 65)
print("STEP 1: Loading df_model.csv")
print("=" * 65)

df = pd.read_csv('df_model.csv')

# Drop non-feature columns
drop_cols = [c for c in ['Unnamed: 0', 'rid', 'viscode'] if c in df.columns]
df = df.drop(columns=drop_cols)

X = df.drop(columns=['converted'])
y = df['converted']

print(f"\nFeature columns ({len(X.columns)}):")
for c in X.columns:
    print(f"  • {c}")

print(f"\nClass distribution:")
print(f"  Class 0 (no conversion): {(y == 0).sum()}")
print(f"  Class 1 (converted):     {(y == 1).sum()}")
print(f"  Total samples:           {len(y)}")
print(f"  Dataset shape:           {X.shape}")

feature_names = list(X.columns)
X_arr = X.values
y_arr = y.values


# ============================================================
# Helper: create a fresh StackingCVClassifier
# ============================================================
def make_fresh_clf(name):
    if name == 'RandomForest':
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif name == 'DecisionTree':
        return DecisionTreeClassifier(random_state=42)
    elif name == 'XGBoost':
        return xgb.XGBClassifier(n_estimators=100, random_state=42,
                                  eval_metric='logloss', verbosity=0, use_label_encoder=False)
    elif name == 'LightGBM':
        return lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    elif name == 'ExtraTrees':
        return ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif name == 'LDA':
        return LinearDiscriminantAnalysis()


def make_stacking(base_names, cv_folds=10):
    base_clfs = [make_fresh_clf(n) for n in base_names]
    meta = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    return StackingCVClassifier(
        classifiers=base_clfs,
        meta_classifier=meta,
        cv=cv_folds,
        use_probas=True,
        use_features_in_secondary=False,
        random_state=42,
        shuffle=True
    )


clf_names = ['RandomForest', 'DecisionTree', 'XGBoost', 'LightGBM', 'ExtraTrees', 'LDA']

# ============================================================
# STEP 2: Standard Model — CV=10
# ============================================================
print("\n" + "=" * 65)
print("STEP 2: Standard Model (all 6 classifiers, CV=10)")
print("=" * 65)
t0 = time.time()

skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
std_model_cv10 = make_stacking(clf_names, cv_folds=10)
std_acc10 = cross_val_score(std_model_cv10, X_arr, y_arr, cv=skf10, scoring='accuracy')
std_auc10 = cross_val_score(make_stacking(clf_names, 10), X_arr, y_arr, cv=skf10, scoring='roc_auc')

print(f"  Accuracy : {std_acc10.mean()*100:.2f}% ± {std_acc10.std()*100:.2f}%")
print(f"  AUC-ROC  : {std_auc10.mean():.4f} ± {std_auc10.std():.4f}")
print(f"  Time     : {time.time()-t0:.1f}s")

# Fit final model on full data and save
std_cv10_final = make_stacking(clf_names, cv_folds=10)
std_cv10_final.fit(X_arr, y_arr)
with open('standard_model_cv10.pkl', 'wb') as f:
    pickle.dump(std_cv10_final, f)
print("  Saved    : standard_model_cv10.pkl ✓")


# ============================================================
# STEP 3: Proposed Model — CV=10 (best combination search)
# ============================================================
print("\n" + "=" * 65)
print("STEP 3: Combination search for Proposed Model (CV=10)")
print("=" * 65)

all_combos = []
for r in range(2, 7):
    for combo in itertools.combinations(clf_names, r):
        all_combos.append(list(combo))

print(f"  Total combinations to test: {len(all_combos)}")
print("  (Using 5-fold outer CV + 3-fold inner stacking for speed)\n")

# Use 5-fold outer and 3-fold inner for the search (fast)
search_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_acc_search = -1
best_combo_cv10 = None
combo_results = []

for i, combo in enumerate(all_combos):
    try:
        model = make_stacking(combo, cv_folds=3)  # fast inner cv during search
        scores = cross_val_score(model, X_arr, y_arr, cv=search_cv, scoring='accuracy')
        mean_acc = scores.mean()
        combo_results.append((mean_acc, scores.std(), combo))
        status = "★" if mean_acc > best_acc_search else " "
        if mean_acc > best_acc_search:
            best_acc_search = mean_acc
            best_combo_cv10 = combo
        print(f"  {status} [{i+1:2d}/{len(all_combos)}] {'+'.join(combo):<60} → {mean_acc*100:.2f}%")
    except Exception as e:
        print(f"    [{i+1}] ERROR: {combo} — {e}")

print(f"\n  ✅ Winning combination CV=10: {best_combo_cv10}")
print(f"     Search accuracy: {best_acc_search*100:.2f}%")

# Now evaluate best combo with proper 10-fold CV
t0 = time.time()
prop_cv10 = make_stacking(best_combo_cv10, cv_folds=10)
prop_acc10 = cross_val_score(prop_cv10, X_arr, y_arr, cv=skf10, scoring='accuracy')
prop_auc10 = cross_val_score(make_stacking(best_combo_cv10, 10), X_arr, y_arr, cv=skf10, scoring='roc_auc')
print(f"  Accuracy (CV=10) : {prop_acc10.mean()*100:.2f}% ± {prop_acc10.std()*100:.2f}%")
print(f"  AUC-ROC  (CV=10) : {prop_auc10.mean():.4f} ± {prop_auc10.std():.4f}")
print(f"  Time             : {time.time()-t0:.1f}s")

# Fit final model on full data and save
prop_cv10_final = make_stacking(best_combo_cv10, cv_folds=10)
prop_cv10_final.fit(X_arr, y_arr)
with open('proposed_model_cv10.pkl', 'wb') as f:
    pickle.dump(prop_cv10_final, f)
print("  Saved            : proposed_model_cv10.pkl ✓")


# ============================================================
# STEP 4: Repeat for CV=5
# ============================================================
print("\n" + "=" * 65)
print("STEP 4a: Standard Model (all 6 classifiers, CV=5)")
print("=" * 65)
t0 = time.time()

skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
std_acc5 = cross_val_score(make_stacking(clf_names, 5), X_arr, y_arr, cv=skf5, scoring='accuracy')
std_auc5 = cross_val_score(make_stacking(clf_names, 5), X_arr, y_arr, cv=skf5, scoring='roc_auc')
print(f"  Accuracy : {std_acc5.mean()*100:.2f}% ± {std_acc5.std()*100:.2f}%")
print(f"  AUC-ROC  : {std_auc5.mean():.4f} ± {std_auc5.std():.4f}")
print(f"  Time     : {time.time()-t0:.1f}s")

std_cv5_final = make_stacking(clf_names, cv_folds=5)
std_cv5_final.fit(X_arr, y_arr)
with open('standard_model_cv5.pkl', 'wb') as f:
    pickle.dump(std_cv5_final, f)
print("  Saved    : standard_model_cv5.pkl ✓")

print("\n" + "=" * 65)
print("STEP 4b: Combination search for Proposed Model (CV=5)")
print("=" * 65)

best_acc5_search = -1
best_combo_cv5 = None

for i, combo in enumerate(all_combos):
    try:
        model = make_stacking(combo, cv_folds=3)
        scores = cross_val_score(model, X_arr, y_arr, cv=skf5, scoring='accuracy')
        mean_acc = scores.mean()
        status = "★" if mean_acc > best_acc5_search else " "
        if mean_acc > best_acc5_search:
            best_acc5_search = mean_acc
            best_combo_cv5 = combo
        print(f"  {status} [{i+1:2d}/{len(all_combos)}] {'+'.join(combo):<60} → {mean_acc*100:.2f}%")
    except Exception as e:
        print(f"    [{i+1}] ERROR: {combo} — {e}")

print(f"\n  ✅ Winning combination CV=5: {best_combo_cv5}")

t0 = time.time()
prop_cv5 = make_stacking(best_combo_cv5, cv_folds=5)
prop_acc5 = cross_val_score(prop_cv5, X_arr, y_arr, cv=skf5, scoring='accuracy')
prop_auc5 = cross_val_score(make_stacking(best_combo_cv5, 5), X_arr, y_arr, cv=skf5, scoring='roc_auc')
print(f"  Accuracy (CV=5) : {prop_acc5.mean()*100:.2f}% ± {prop_acc5.std()*100:.2f}%")
print(f"  AUC-ROC  (CV=5) : {prop_auc5.mean():.4f} ± {prop_auc5.std():.4f}")
print(f"  Time            : {time.time()-t0:.1f}s")

prop_cv5_final = make_stacking(best_combo_cv5, cv_folds=5)
prop_cv5_final.fit(X_arr, y_arr)
with open('proposed_model_cv5.pkl', 'wb') as f:
    pickle.dump(prop_cv5_final, f)
print("  Saved           : proposed_model_cv5.pkl ✓")


# ============================================================
# STEP 5: SHAP Plots
# ============================================================
print("\n" + "=" * 65)
print("STEP 5: Generating SHAP plots (KernelExplainer)...")
print("=" * 65)


def generate_shap_beeswarm(model, X_data, feat_names, title, filename, n_bg=50, n_explain=150):
    print(f"  Computing SHAP for: {filename}")
    X_np = X_data if isinstance(X_data, np.ndarray) else X_data.values

    # Background dataset (k-means compressed)
    bg = shap.kmeans(X_np, min(n_bg, len(X_np)))
    explainer = shap.KernelExplainer(model.predict_proba, bg)

    # Explain a subset for speed
    X_explain = X_np[:n_explain]
    shap_vals = explainer.shap_values(X_explain, nsamples=80)

    # For binary: take class 1 shap values
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    # Build figure
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    shap.summary_plot(sv, X_explain, feature_names=feat_names,
                      show=False, plot_type='dot', color_bar=True)

    # Styling
    fig = plt.gcf()
    fig.set_facecolor('white')
    fig.set_size_inches(10, 8)
    for ax in fig.axes:
        ax.set_facecolor('white')
        ax.tick_params(labelsize=11)

    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("SHAP value (impact on model output)", fontsize=12)
    plt.ylabel("Features", fontsize=12)

    # Update colorbar label
    for ax in fig.axes:
        if ax.get_ylabel() in ('', 'Feature value'):
            ax.set_ylabel("Feature value", fontsize=11)
        # Set colorbar ticks if it's the colorbar axis
        if hasattr(ax, 'collections') and len(ax.collections) == 0:
            try:
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['Low', 'High'])
            except Exception:
                pass

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f"  Saved: {filename} ✓")


generate_shap_beeswarm(
    prop_cv10_final, X_arr, feature_names,
    "SHAP Summary Plot — aLZfusion Proposed Model (CV=10)",
    "shap_cv10.png"
)

generate_shap_beeswarm(
    prop_cv5_final, X_arr, feature_names,
    "SHAP Summary Plot — aLZfusion Proposed Model (CV=5)",
    "shap_cv5.png"
)


# ============================================================
# STEP 6: Accuracy bar charts
# ============================================================
print("\n" + "=" * 65)
print("STEP 6: Generating accuracy bar charts...")
print("=" * 65)


def plot_accuracy_bar(std_mean, std_std, prop_mean, prop_std, cv_label, filename):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    ax.set_facecolor('white')

    labels = ['Standard Model', 'Proposed Model']
    accs = [std_mean * 100, prop_mean * 100]
    stds = [std_std * 100, prop_std * 100]
    colors = ['steelblue', 'mediumseagreen']
    x = np.arange(len(labels))

    bars = ax.bar(x, accs, color=colors, width=0.5,
                  yerr=stds, capsize=10,
                  error_kw={'linewidth': 2.0, 'ecolor': '#222222', 'capthick': 2.0},
                  zorder=3)

    # Value labels on bars
    for bar, acc, std_v in zip(bars, accs, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std_v + 0.8,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold', color='#111111')

    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(f"Accuracy Comparison — {cv_label}", fontsize=15, fontweight='bold', pad=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=11)

    # Legend patch
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='Standard Model'),
                       Patch(facecolor='mediumseagreen', label='Proposed Model')]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename} ✓")


plot_accuracy_bar(std_acc10.mean(), std_acc10.std(),
                  prop_acc10.mean(), prop_acc10.std(),
                  "CV=10", "accuracy_cv10.png")

plot_accuracy_bar(std_acc5.mean(), std_acc5.std(),
                  prop_acc5.mean(), prop_acc5.std(),
                  "CV=5", "accuracy_cv5.png")


# ============================================================
# STEP 7: Summary table
# ============================================================
print("\n" + "=" * 65)
print("STEP 7: Final Summary Table")
print("=" * 65)

header = f"{'Model':<18} {'CV':>5} {'Acc (%)':>10} {'Std':>8} {'AUC':>8}   Best Classifier Combination"
print(header)
print("-" * 100)

rows = [
    ("Standard Model", "CV=10",
     f"{std_acc10.mean()*100:.2f}", f"{std_acc10.std()*100:.2f}", f"{std_auc10.mean():.4f}",
     "All 6 classifiers"),
    ("Proposed Model", "CV=10",
     f"{prop_acc10.mean()*100:.2f}", f"{prop_acc10.std()*100:.2f}", f"{prop_auc10.mean():.4f}",
     " + ".join(best_combo_cv10)),
    ("Standard Model", "CV=5",
     f"{std_acc5.mean()*100:.2f}", f"{std_acc5.std()*100:.2f}", f"{std_auc5.mean():.4f}",
     "All 6 classifiers"),
    ("Proposed Model", "CV=5",
     f"{prop_acc5.mean()*100:.2f}", f"{prop_acc5.std()*100:.2f}", f"{prop_auc5.mean():.4f}",
     " + ".join(best_combo_cv5)),
]

for r in rows:
    print(f"{r[0]:<18} {r[1]:>5} {r[2]:>10} {r[3]:>8} {r[4]:>8}   {r[5]}")

print("\n" + "=" * 65)
print("✅ All steps complete!")
print("=" * 65)

saved = [
    'standard_model_cv10.pkl', 'proposed_model_cv10.pkl',
    'standard_model_cv5.pkl',  'proposed_model_cv5.pkl',
    'shap_cv10.png', 'shap_cv5.png',
    'accuracy_cv10.png', 'accuracy_cv5.png'
]
print("\nFiles saved:")
for f in saved:
    print(f"  📄 {f}")
