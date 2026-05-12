import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# LOAD THE DATA
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

df = pd.DataFrame(x, columns=cancer.feature_names)
df['tumor type'] = pd.Categorical.from_codes(y, cancer.target_names)
print(df)

print(df.head())

# SCALE THE DATA
scaled_data = StandardScaler().fit_transform(x)

# PCA DATA
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

# SCRER PLOT
var_per = pca.explained_variance_ratio_
cum_var_per = var_per.cumsum()
labels = ["PC" + str(i) for i in range(1,31)]
plt.bar(range(1,31), height=cum_var_per ,tick_label=labels)
plt.xlabel('Principal Component')
plt.ylabel('Variance') 
plt.title('Scree Plot')
plt.show()

# According to the scree plot, k=7

# PCA WITH k=7
pca = PCA(n_components=7)
new_pca = pca.fit_transform(scaled_data)

# NEW PCA PLOT
plt.scatter()