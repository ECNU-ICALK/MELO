import pickle
import torch
import pandas as pd
import plotly.express as px
base_dir = 'logs/VecDB'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import transformers
with open(f'{base_dir}/T5Small_100_e75/log.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
sns.set(style="darkgrid")

tokenizer = transformers.AutoTokenizer.from_pretrained('google/t5-small-ssm-nq')
vecdb = loaded_dict['all_VecDB']
import time
clusters = vecdb.table
X = []
Y = []
for index, cluster in enumerate(clusters):

    if len(cluster['points']) > 3 and cluster['radius']:
        key_label = cluster['key_label']
        key_label = key_label.masked_fill(key_label == -100, tokenizer.pad_token_id)
        key_label = tokenizer.decode(key_label, skip_special_tokens=True)
        print(key_label,index)

        for point in cluster['points']:
            X.append(point.get_key())
            Y.append(index)

X = torch.stack(X,dim=0).cpu().numpy()
y = torch.tensor(Y).cpu().numpy()

# fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=Y, opacity=0.8)
# fig.show()

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))

X, y = None, None

print('Size of the dataframe: {}'.format(df.shape))

pca = PCA(n_components=30)
pca_result = pca.fit_transform(df[feat_cols].values)

df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,4]
# df['pca-three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
np.random.seed(0)

rndperm = np.random.permutation(df.shape[0])
plt.figure(figsize=(5,5), dpi = 800)
ax = sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("tab10"),
    legend=False,
    data=df,
    marker= 'o',
    s=120,
)
ax.set(xlabel=None)
ax.set(ylabel=None)
plt.tight_layout()
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('./ablation/ablation_res/pca.jpg')



