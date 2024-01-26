import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import math
import numpy as np
import umap
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import time
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# 加载 CIFAR-10 训练集
trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
all_images = []
all_labels = []
# 遍历数据集并收集所有图像和标签
for img, label in trainset:
    all_images.append(img)
    all_labels.append(label)

# 可选：将列表转换为 PyTorch 张量
# 使用 torch.stack 而不是 torch.tensor，以保持图像的张量属性
all_images_tensor = torch.stack(all_images).numpy()
all_labels_tensor = torch.tensor(all_labels)
n_images, height, width, channels = all_images_tensor.shape
flattened_images = all_images_tensor.reshape((n_images, height * width * channels))
print(flattened_images.shape)
from sklearn.neighbors import NearestNeighbors
import numpy as np
# Function to calculate RRE
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import numpy as np


class EvaluationMetrics:
    def __init__(self, high_dim_data, low_dim_data, k=10):
        self.high_dim_data = high_dim_data
        self.low_dim_data = low_dim_data
        self.k = k
        self.distance_input = pairwise_distances(high_dim_data)
        self.distance_latent = pairwise_distances(low_dim_data)
        self.neighbour_input, self.rank_input = self._neighbours_and_ranks(self.distance_input)
        self.neighbour_latent, self.rank_latent = self._neighbours_and_ranks(self.distance_latent)
    def _neighbours_and_ranks(self, distances):
        indices = np.argsort(distances, axis=-1, kind="stable")
        neighbourhood = indices[:, 1:self.k+1]
        ranks = indices.argsort(axis=-1, kind="stable")
        return neighbourhood, ranks
    def E_mrre(self):
        n = self.high_dim_data.shape[0]
        mrre_ZX, mrre_XZ = 0.0, 0.0
        for row in range(n):
            for neighbour in self.neighbour_latent[row]:
                rx = self.rank_input[row, neighbour]
                rz = self.rank_latent[row, neighbour]
                if rz != 0:  # Avoid division by zero
                    mrre_ZX += abs(rx - rz) / rz
        for row in range(n):
            for neighbour in self.neighbour_input[row]:
                rx = self.rank_input[row, neighbour]
                rz = self.rank_latent[row, neighbour]
                if rx != 0:  # Avoid division by zero
                    mrre_XZ += abs(rx - rz) / rx
        C = n * sum([abs(2 * j - n - 1) / j for j in range(1, self.k+1)])
        return mrre_ZX / C, mrre_XZ / C
original_data = flattened_images
all_labels=all_labels
print(original_data.shape)
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
start_time=time.time()
# tsne = TSNE(n_components=2)
tsne = umap.UMAP(n_components=2)
reduced_data = tsne.fit_transform(original_data)
X_tsne = reduced_data
print("Time:",time.time()-start_time)
# Calculate RRE
high_dim_data = original_data
low_dim_data = reduced_data
labels = all_labels
print("YES")
evaluation_metrics = EvaluationMetrics(high_dim_data, low_dim_data, k=10)
rre = evaluation_metrics.E_mrre()
print("YES")
print(f"Mean Relative Rank Error (MRRE): {(rre[0]+rre[1])/2}")
# 应用 t-SNE
# 可视化结果
clf = SVC(kernel='linear')  # 可以选择不同的核函数

# 训练模型
v=int(math.floor(X_tsne.size*0.2))
clf.fit(X_tsne[v:], all_labels[v:])
# 预测
y_pred = clf.predict(X_tsne[:v])

# 评估模型
accuracy = accuracy_score(all_labels[:v], y_pred)
print("Accuracy:", format(accuracy,'.3f'))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=all_labels, s=1)
plt.colorbar()
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE feature 1")
plt.ylabel("t-SNE feature 2")
plt.show()