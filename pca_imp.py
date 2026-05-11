import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ============================================================
# PART 1: HEIGHT + WEIGHT — sklearn way
# ============================================================

n = 200
body_size = np.random.randn(n)
height_cm = 170 + body_size * 8  + np.random.randn(n) * 2
weight_kg =  70 + body_size * 7  + np.random.randn(n) * 3
X_raw = np.column_stack([height_cm, weight_kg])

print("=" * 55)
print("PART 1: Height + Weight")
print("=" * 55)

# the sklearn workflow is always:
#   1. scale  →  2. fit PCA  →  3. transform  →  4. inspect
#
# StandardScaler does the centering (and scaling to unit variance)
# so you don't do X - X.mean() manually anymore
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
# fit_transform(X_raw) is equivalent to:
#   scaler.fit(X_raw)       ← learns mean and std from data
#   scaler.transform(X_raw) ← applies (x - mean) / std to each feature

pca = PCA(n_components=2)
pca.fit(X_scaled)
# fit() computes the principal components but doesn't transform yet
# it populates: pca.components_, pca.explained_variance_ratio_

X_pca = pca.transform(X_scaled)
# transform() projects the data onto the PC axes
# equivalent to X_scaled @ pca.components_.T

# or do both at once:  X_pca = pca.fit_transform(X_scaled)

print(f"\nExplained variance ratio: {pca.explained_variance_ratio_.round(3)}")
print(f"PC1: {pca.explained_variance_ratio_[0]:.1%} of variance")
print(f"PC2: {pca.explained_variance_ratio_[1]:.1%} of variance")

# pca.components_ is the same as our eigenvectors matrix from scratch
# shape: (n_components, n_features)
# each ROW is a PC direction (sklearn convention — from scratch it was columns)
print(f"\nPC directions (pca.components_):")
print(f"  PC1: height={pca.components_[0, 0]:.3f}, weight={pca.components_[0, 1]:.3f}")
print(f"  PC2: height={pca.components_[1, 0]:.3f}, weight={pca.components_[1, 1]:.3f}")

# reconstruct from PC1 only — inverse_transform does the un-projection
X_1d = X_pca[:, [0]]                          # keep only PC1 scores
X_1d_padded = np.hstack([X_1d, np.zeros((n, 1))])  # zero out PC2
X_recon_scaled = pca.inverse_transform(X_1d_padded) # back to scaled space
X_recon = scaler.inverse_transform(X_recon_scaled)  # back to original units (cm, kg)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle('Part 1: Height + Weight — sklearn PCA', fontsize=13)

axes[0].scatter(height_cm, weight_kg, alpha=0.4, s=20, color='steelblue')
center = X_raw.mean(axis=0)
for i, (color, label) in enumerate(zip(['tomato', 'seagreen'], ['PC1', 'PC2'])):
    # pca.components_[i] is in scaled space — un-scale the direction for plotting
    v_scaled = pca.components_[i] * np.sqrt(pca.explained_variance_[i]) * 3
    v = v_scaled * scaler.scale_  # convert back to original units
    axes[0].annotate('', xy=center + v, xytext=center - v,
                     arrowprops=dict(arrowstyle='<->', color=color, lw=2.5))
    axes[0].text(center[0]+v[0]*1.2, center[1]+v[1]*1.2,
                 label, color=color, fontsize=10, fontweight='bold', ha='center')
axes[0].set_xlabel('Height (cm)'); axes[0].set_ylabel('Weight (kg)')
axes[0].set_title('Raw data + PC directions')

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4, s=20, color='steelblue')
axes[1].axhline(0, color='grey', lw=0.5); axes[1].axvline(0, color='grey', lw=0.5)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[1].set_title('Data in PC space')

axes[2].scatter(height_cm, weight_kg, alpha=0.25, s=20,
                color='steelblue', label='original')
axes[2].scatter(X_recon[:, 0], X_recon[:, 1], alpha=0.7, s=20,
                color='tomato', label='PC1 only')
for i in range(0, n, 8):
    axes[2].plot([height_cm[i], X_recon[i,0]],
                 [weight_kg[i],  X_recon[i,1]], 'grey', lw=0.5, alpha=0.5)
axes[2].set_xlabel('Height (cm)'); axes[2].set_ylabel('Weight (kg)')
axes[2].set_title(f'Reconstructed from PC1 only\n({pca.explained_variance_ratio_[0]:.1%} kept)')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('sklearn_height_weight.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved sklearn_height_weight.png")


# ============================================================
# PART 2: 4 FEATURES — sklearn way
# Height, weight, arm span, shoe size
# ============================================================

print("\n" + "=" * 55)
print("PART 2: Height, Weight, Arm Span, Shoe Size")
print("=" * 55)

body_size2  = np.random.randn(n)
height2     = 170 + body_size2 * 8   + np.random.randn(n) * 2
weight2     =  70 + body_size2 * 7   + np.random.randn(n) * 4
arm_span    = 171 + body_size2 * 7.5 + np.random.randn(n) * 3
shoe_size   =  42 + body_size2 * 1.8 + np.random.randn(n) * 0.8
X4_raw      = np.column_stack([height2, weight2, arm_span, shoe_size])
feature_names = ['Height', 'Weight', 'Arm span', 'Shoe size']

# scale — critical here because features have different units
# (cm vs kg vs EU shoe sizes)
scaler4  = StandardScaler()
X4_scaled = scaler4.fit_transform(X4_raw)

# fit PCA with all 4 components first to see variance breakdown
pca4_full = PCA()               # no n_components = keep all
pca4_full.fit(X4_scaled)

print("\nVariance explained per component:")
for i, v in enumerate(pca4_full.explained_variance_ratio_):
    bar = '█' * int(v * 40)
    print(f"  PC{i+1}: {v:.1%}  {bar}")
print(f"\nCumulative: {np.cumsum(pca4_full.explained_variance_ratio_).round(3)}")

# now fit with just the components we want to keep
# option A: specify exact number
pca4 = PCA(n_components=2)

# option B: let sklearn pick automatically (keep enough for 95% variance)
pca4_auto = PCA(n_components=0.95)
pca4_auto.fit(X4_scaled)
print(f"\nComponents needed for 95% variance: {pca4_auto.n_components_}")
# for this data it should say 1, because everything is driven by body size

X4_pca = pca4.fit_transform(X4_scaled)
print(f"\nOriginal shape: {X4_scaled.shape}  →  Reduced shape: {X4_pca.shape}")

# reconstruct from PC1 only using inverse_transform
X4_1pc       = np.hstack([X4_pca[:, [0]], np.zeros((n, 1))])  # zero out PC2
X4_recon_sc  = pca4.inverse_transform(X4_1pc)
X4_recon     = scaler4.inverse_transform(X4_recon_sc)

print(f"\nReconstruction MSE from PC1 only:")
mse = np.mean((X4_raw - X4_recon)**2, axis=0)
for fname, err in zip(feature_names, mse):
    print(f"  {fname:<12}: {err:.2f}")

# --- plots for part 2 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle('Part 2: 4 features — sklearn PCA', fontsize=13)

# scree plot
var = pca4_full.explained_variance_ratio_
cum = np.cumsum(var)
axes[0].bar(range(1, 5), var * 100, color='steelblue', alpha=0.8, edgecolor='white')
axes[0].plot(range(1, 5), cum * 100, 'o-', color='tomato', lw=2, label='cumulative')
axes[0].axhline(90, color='seagreen', linestyle='--', lw=1, label='90%')
axes[0].set_xlabel('Component'); axes[0].set_ylabel('Variance explained (%)')
axes[0].set_title('Scree plot')
axes[0].set_xticks([1, 2, 3, 4]); axes[0].set_ylim(0, 110)
axes[0].legend(fontsize=8)
for i, v in enumerate(var):
    axes[0].text(i+1, v*100+2, f'{v:.0%}', ha='center', fontsize=9)

# loadings heatmap — pca.components_ rows=PCs, cols=features
loadings = pca4.components_  # shape (2, 4)
im = axes[1].imshow(loadings, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(feature_names, fontsize=9)
axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(['PC1', 'PC2'])
axes[1].set_title('Loadings (pca.components_)\nwhat each PC is made of')
for i in range(2):
    for j in range(4):
        axes[1].text(j, i, f'{loadings[i,j]:.2f}', ha='center', va='center',
                     fontsize=10, color='white' if abs(loadings[i,j]) > 0.5 else 'black')
plt.colorbar(im, ax=axes[1], fraction=0.046)

# PC1 score vs each original feature
pc1_scores = X4_pca[:, 0]
colors = ['steelblue', 'tomato', 'seagreen', 'mediumpurple']
for i, (fname, color) in enumerate(zip(feature_names, colors)):
    axes[2].scatter(pc1_scores, X4_raw[:, i] / X4_raw[:, i].max(),
                    alpha=0.3, s=10, color=color, label=fname)
axes[2].set_xlabel('PC1 score ("body size")')
axes[2].set_ylabel('Normalised feature value')
axes[2].set_title('PC1 vs all features\n(one score summarises everything)')
axes[2].legend(fontsize=8, markerscale=2)

plt.tight_layout()
plt.savefig('sklearn_4features.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved sklearn_4features.png")

print("\n" + "=" * 55)
print("SCRATCH vs SKLEARN — what changed")
print("=" * 55)
print("Scratch:  X - X.mean()              → StandardScaler().fit_transform()")
print("Scratch:  (X.T @ X) / (n-1)         → handled internally by sklearn")
print("Scratch:  np.linalg.eigh()           → sklearn uses SVD (more stable)")
print("Scratch:  X @ eigenvectors           → pca.transform(X)")
print("Scratch:  X_1d @ eigenvectors.T      → pca.inverse_transform()")
print("Scratch:  eigenvalues/eigenvalues.sum → pca.explained_variance_ratio_")
print("Scratch:  eigenvectors[:, i]         → pca.components_[i]  (note: rows not cols)")
