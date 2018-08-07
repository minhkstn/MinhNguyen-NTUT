"""
Khong biet label cua cac diem du lieu, tu chia thanh cac clustering
Tóm tắt thuật toán
Đầu vào: matran dữ liệu X (dxN) - N phần tử, mỗi phần tử ứng vs 1 cột. và số lượng cluster cần tìm (K< K)
Đầu ra: matran các centroid(dxK)- K centroid, mỗi centroid ứng vs 1 cột. Ma trận label Y(NxK) - 1 label = 1 hang
    1. chọn K điểm bất kỳ trong training set làm centroid ban đầu
    2. Phân mỗi điển dữ liệu vào cluster có centroid gần nh
    3. Nếu kết quả bước 2 k thay đổi so vs vòng lặp trc đó ==> dừng thuật toán
    4. Cập nhật centroid cho từng cluster = cách lấy trung bình tọa độ các phần tử đã đc gán cho cluster ở bước 2
    5. quay lại bước 2

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import random

np.random.seed(18)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(mean=means[0], cov=cov,size=N)   #?
X1 = np.random.multivariate_normal(mean=means[1], cov=cov,size=N)
X2 = np.random.multivariate_normal(mean=means[2], cov=cov,size=N)

X = np.concatenate((X0, X1, X2), axis=0)    #?
K = 3 # 3 clusters
original_label = np.asarray([0]*N + [1]*N + [2]*N).T    #?

"""
Các hàm cần thiết cho K-mean clustering
    kmeans_init_centroids: khởi tạo các centroids ban đầu
    kmeans_assign_labels: tìm label mới cho các điểm khi cố định centroid
    kmeans_update_centroids: cập nhật các centroid khi biết label của mỗi điểm dữ liệu
    has_converged: kiểm tra điều kiện dừng của thuật toán
"""
def kmeans_init_centroids(X, k):
    # randomly pick k rows of X as initial centroids
    return X[np.random.choice(X.shape[0], k, replace=False)] #?

def kmeans_assign_labels(X, centroids):
    # calculate pairwise distance btw data and centroids
    D = cdist(X, centroids) #?
    # return index of the closest centroid
    return np.argmin(D, axis=1)

def has_converged(centroids, new_centroids):
    # return True if two sets of centroids are the same
    return (set([tuple(a) for a in centroids]) ==
            set([tuple(a) for a in new_centroids])) #?

def kmeans_update_centroids(X, labels, K):
    centroids = np.zeros((K, X.shape[1]))   # K hàng và X.shape[1] cột (= số cột của X)
    for k in range(K):
        # collect all points that are assigned to the k-th cluster
        Xk = X[labels == k, :]
        centroids[k, :] = np.mean(Xk, axis=0)
    return centroids

# main part
def kmeans(X, K):
    centroids = [kmeans_init_centroids(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centroids[-1]))  # centroid cuoi cung trong array centroids
        new_centroids = kmeans_update_centroids(X, labels[-1], K)
        if has_converged(centroids[-1], new_centroids):
            break
        centroids.append(new_centroids)
        it += 1
    return (centroids, labels, it)

######################################
(centroids, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:\n', centroids[-1])
print('iterations:', it)
# kmeans_display(X,labels[-1])
