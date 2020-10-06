import numpy as np
from hct66 import generate_data, get_feature
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 特征标准，降维
def process_data(X, n_components=2):
    X = [get_feature(x).detach().numpy()[0] for x in X]
    X = np.array(X)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    return pca.fit(X).transform(X)

# 欧氏距离
def uclidean(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

class KMeasPP:
    def __init__(self, n_clusters=3, max_iter=2):
        assert n_clusters >= 2
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    # 初始化中心点
    def _init_centers(self, X):
        cnt, _ = X.shape
        distances = np.zeros((cnt, cnt))
        # 计算所以点之间的两两距离
        for i in range(cnt):
            for j in range(cnt):
                if i != j:
                    distances[i, j] = uclidean(X[i], X[j])
        # 两两间距离最大的index
        idx = np.argmax(distances)
        # 距离最大的两个点的index
        p_index = [int(idx / 10), idx % 10]
        # 寻找其余的中心点：距离已选择的中心点距离最远的点
        for _ in range(2, self.n_clusters):
            t = np.sum(distances[p_index], axis=0)
            t[p_index] = -1
            p_index.append(np.argmax(t))
        # 返回初始化中心点
        return [X[i] for i in p_index]

    # 更新中心点
    def _update_centers(self, centers, cat, X):
        for i in range(self.n_clusters):
            centers[i] = (np.mean(X[np.where(cat == i)], axis=0))
        return centers

    # 所有点距离中心点的距离
    def _get_distances(self, centers, X):
        cnt = X.shape[0]
        distances = np.zeros((self.n_clusters, cnt))
        for i in range(self.n_clusters):
            for j in range(cnt):
                distances[i, j] = uclidean(centers[i], X[j])
        return distances

    # 拟合并预测
    def fit_predict(self, X):
        # 初始化中心点
        centers = self._init_centers(X)
        for i in range(self.max_iter):
            distances = self._get_distances(centers, X)
            # 每个样本所属类别
            cat = np.argmin(distances, axis=0)
            centers = self._update_centers(centers, cat, X)
            # print("iter: {}/{}, centers:{}".format(i+1, self.max_iter, centers))
        return cat

N_CLUSTERS = 3
data, label = generate_data()
X = process_data(data)
model = KMeasPP(n_clusters=N_CLUSTERS)
p = model.fit_predict(X)
print("label:", label)
print("类别 : ", list(p))

colors = {0: 'red', 1: 'green', 2: 'blue'}
catalog = {0: 'catalog-1', 1: 'catalog-2', 2: 'catalog-3'}
for i in range(N_CLUSTERS):
    x_cat = X[np.where(p == i)]
    x, y = zip(*[[x[0], x[1]] for x in x_cat])
    plt.scatter(x, y, c=colors[i], label=catalog[i])

plt.legend()
plt.show()
