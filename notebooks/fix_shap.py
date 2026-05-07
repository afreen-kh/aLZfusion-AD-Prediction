import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

def fix_shap(pkl_path, output_path, title):
    print(f'Starting {pkl_path}...')
    if not os.path.exists(pkl_path):
        print(f'Error: {pkl_path} not found.')
        return
    
    # 1. Load Data
    df = pd.read_csv('df_model.csv')
    drop_cols = [c for c in ['Unnamed: 0', 'rid', 'viscode'] if c in df.columns]
    X = df.drop(columns=drop_cols + ['converted'])
    
    # 4. Rename Columns
    rename_dict = {
        'adas13': 'ADAS-13 Score',
        'cdrsb': 'CDR-SB Score',
        'mmse': 'MMSE Score',
        'age': 'Age',
        'apoe4': 'APOE4 Alleles',
        'pteducat': 'Education',
        'ptgender_Female': 'Gender (Female)',
        'ptgender_Male': 'Gender (Male)',
        'x_hippocampus_l': 'L Hippocampus Volume',
        'x_hippocampus_r': 'R Hippocampus Volume',
        'x_entorhinal_l': 'L Entorhinal Volume',
        'x_entorhinal_r': 'R Entorhinal Volume',
        'x_entorhinal_l_thick': 'L Entorhinal Thickness',
        'x_entorhinal_r_thick': 'R Entorhinal Thickness',
        'diff_1': 'Longitudinal Diff',
        'dx_bl_CN': 'Baseline CN',
        'dx_bl_LMCI': 'Baseline LMCI'
    }
    X.columns = [rename_dict.get(c, c) for c in X.columns]
    
    # 2. Load Model
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)
    print(f'  Model loaded: {type(model).__name__}')
    
    # 5. Explainer Selection
    best_component = None
    try:
        # For mlxtend StackingCVClassifier, fitted base classifiers are in clfs_
        clfs = getattr(model, 'clfs_', [])
        for c in clfs:
            if 'XGB' in str(type(c)) or 'Forest' in str(type(c)) or 'Extra' in str(type(c)):
                best_component = c
                print(f'  Selected tree component: {type(c).__name__}')
                break
    except Exception as e:
        print(f'  Could not extract components: {e}')
        
    if not best_component:
        # Fallback to meta if it's a tree
        best_component = getattr(model, 'meta_clf_', None)
        if best_component:
            print(f'  Using meta-classifier: {type(best_component).__name__}')
        else:
            print('  Error: Could not find any fitted component!')
            return

    print('  Computing SHAP values (TreeExplainer)...')
    explainer = shap.TreeExplainer(best_component)
    
    # Explain all features
    shap_values = explainer.shap_values(X)
    
    # 6. Single class extraction
    # If list, it's [class 0, class 1]. We want class 1.
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # 8. Plot Parameters
    plt.figure(figsize=(10, 8), facecolor='white')
    
    # 7. Rank and Plot
    # 9. Single x-axis
    shap.summary_plot(sv, X, plot_type='dot', show=False, color_bar=True)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('SHAP value (impact on model output)', fontsize=12)
    
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    fig.set_facecolor('white')
    
    # Fix Colorbar Label
    for ax in fig.axes:
        if ax.get_ylabel() == 'Feature value':
            ax.set_yticklabels(['Low', 'High'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Successfully saved: {output_path}')

if __name__ == '__main__':
    fix_shap('proposed_model_cv10.pkl', 'shap_cv10_fixed.png', 'SHAP Summary Plot — aLZfusion Proposed Model (CV=10)')
    fix_shap('proposed_model_cv5.pkl', 'shap_cv5_fixed.png', 'SHAP Summary Plot — aLZfusion Proposed Model (CV=5)')
    print('All tasks done.')
