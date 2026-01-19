import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import directed_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse.linalg import cg
import psutil


def parse_args():
    parser = argparse.ArgumentParser(
        description='Directed Temporal SIRGN with Laplacian Optimization')
    parser.add_argument("--dataset", type=str, default="B4E", choices=["B4E", "MulDiGraph", "TXNT"],
                        help="Dataset to use (default: B4E)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input graph path (default: ./dataset/<dataset>/output_transactions.txt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output embedding path (default: ./dataset/<dataset>/graph_emb.txt)")
    parser.add_argument("--depth", type=int, default=10, help="Number of iterations")
    parser.add_argument("--alpha", type=float, default=1.0, help="Temporal decay factor")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters")
    parser.add_argument("--stop", default=True, action="store_true", help="Stop at convergence")
    parser.add_argument("--beta", type=float, default=10.0, help="Inverse temperature for K-means")
    parser.add_argument("--kmeans_iter", type=int, default=10, help="K-means iterations")
    parser.add_argument("--lambda_weight", type=float, default=1.0, help="Laplacian regularization weight")
    parser.add_argument("--mu_weight", type=float, default=1.0, help="Identity regularization weight")
    args = parser.parse_args()

    if args.input is None:
        args.input = f"./dataset/{args.dataset}/output_transactions.txt"
    if args.output is None:
        args.output = f"./dataset/{args.dataset}/graph_emb.txt"

    return args

class DifferentiableKMeans:
    def __init__(self, n_clusters, beta, max_iter=10):
        self.n_clusters = n_clusters
        self.beta = beta
        self.max_iter = max_iter

    def initialize_centroids(self, embeddings):
        n_samples = embeddings.shape[0]
        centroids = [embeddings[np.random.randint(n_samples)]]
        for _ in range(self.n_clusters - 1):
            distances = torch.cdist(embeddings, torch.stack(centroids), p=2).min(dim=1)[0]
            probs = distances / distances.sum()
            next_centroid_idx = np.random.choice(n_samples, p=probs.numpy())
            centroids.append(embeddings[next_centroid_idx])
        return torch.stack(centroids)

    def fit(self, embeddings):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        centroids = self.initialize_centroids(embeddings)
        for _ in range(self.max_iter):
            cos_sim = F.cosine_similarity(embeddings.unsqueeze(1), centroids.unsqueeze(0), dim=2)
            assignments = torch.softmax(self.beta * cos_sim, dim=1)
            new_centroids = torch.matmul(assignments.t(), embeddings)
            new_centroids = new_centroids / (assignments.sum(dim=0, keepdim=True).t() + 1e-10)
            centroids = F.normalize(new_centroids, p=2, dim=1)
        cos_sim = F.cosine_similarity(embeddings.unsqueeze(1), centroids.unsqueeze(0), dim=2)
        assignments = torch.softmax(self.beta * cos_sim, dim=1)
        return assignments, centroids

def construct_adjacency_matrix(G, nv, alpha):
    """Construct graph Laplacian matrix with temporal decay."""
    row, col, data = [], [], []
    for v in range(nv):
        for (t, lii, lio) in G[v]:
            weight = np.exp(-t / alpha)
            lii_list = lii.tolist() if lii is not None and lii.size > 0 else []
            lio_list = lio.tolist() if lio is not None and lio.size > 0 else []
            for u in lii_list:
                row.append(v)
                col.append(u)
                data.append(weight)
            for u in lio_list:
                row.append(v)
                col.append(u)
                data.append(weight)
    A = sp.csr_matrix((data, (row, col)), shape=(nv, nv))
    A = A + A.T
    A = (A > 0).astype(float)
    D = sp.diags(A.sum(axis=1).A1)
    L = D - A
    return L, A

def construct_cluster_laplacian(A, assignments, n_clusters, nv):
    """Construct cluster-specific Laplacian matrices."""
    cluster_laplacians = []
    for c in range(n_clusters):
        membership = assignments[:, c].reshape(-1, 1)
        M_c = sp.diags(membership.flatten())
        A_c = M_c @ A @ M_c
        A_c = (A_c > 0).astype(float)
        D_c = sp.diags(A_c.sum(axis=1).A1)
        L_c = D_c - A_c
        cluster_laplacians.append(L_c)
    return cluster_laplacians

def laplacian_optimization(embeddings, G, assignments, centroids, alpha, lambda_weight, mu_weight):
    """Optimize embeddings using Laplacian regularization with sparse matrices."""
    nv, d = embeddings.shape
    n_clusters = assignments.shape[1]
    L, A = construct_adjacency_matrix(G, nv, alpha)
    cluster_laplacians = construct_cluster_laplacian(A, assignments, n_clusters, nv)
    L_c_sum = sum(cluster_laplacians)
    I = sp.eye(nv)
    A_matrix = L + lambda_weight * L_c_sum + mu_weight * I
    B_matrix = mu_weight * embeddings
    Z = np.zeros_like(embeddings)
    for i in range(d):
        Z[:, i], _ = cg(A_matrix, B_matrix[:, i], x0=embeddings[:, i], maxiter=100)
    scaler = MinMaxScaler()
    Z = scaler.fit_transform(Z)
    return Z

def dirtemporalAggregation1(embd, G, v, alpha):
    """Aggregate temporal neighbor embeddings for a single node."""
    k = embd.shape[1]
    h = np.zeros((k * 2, k * 2))
    h1 = np.zeros((1, k * 2))
    w = []
    for i in range(len(G[v])):
        (ti, lii, lio) = G[v][i]
        lii_list = lii.tolist() if lii is not None and lii.size > 0 else []
        lio_list = lio.tolist() if lio is not None and lio.size > 0 else []
        wiin = np.zeros((k,))
        wiout = np.zeros((k,))
        for f in lii_list:
            wiin += embd[f, :]
        for g in lio_list:
            wiout += embd[g, :]
        wiboth = np.hstack([wiin, wiout])
        wiboth = wiboth / (np.linalg.norm(wiboth) + 1e-10)
        h1 += wiboth
        w.append(wiboth.reshape((k * 2, 1)))
    z = np.zeros((1, k * 2))
    for i in range(1, len(G[v])):
        (tni, lii, lio) = G[v][i]
        (tnim1, lim1i, lim1o) = G[v][i - 1]
        exp_input = np.clip((tni - tnim1) / alpha, -20, 20)
        z = np.exp(exp_input) * (w[i - 1].transpose() + z)
        z = z / (np.linalg.norm(z) + 1e-10)
        a = w[i] * z
        h += a
    g = h.flatten()
    return np.hstack([g.reshape((1, g.shape[0])), h1])

def dirtemporalAggregation(embd, G, alpha):
    """Aggregate temporal neighbor embeddings for all nodes."""
    m = []
    nv = len(G)
    for v in range(nv):
        m.append(dirtemporalAggregation1(embd, G, v, alpha))
    return np.vstack(m)

def getnumber(emb):
    """Calculate the number of unique embeddings."""
    ss = set()
    for x in range(emb.shape[0]):
        sd = ','.join(str(emb[x, y]) for y in range(emb.shape[1]))
        ss.add(sd)
    return len(ss)

def dirtemporalSirGN(G, n, alpha, iter=10, beta=10.0, kmeans_iter=10, lambda_weight=1.0, mu_weight=1.0):
    """Run SIRGN with Laplacian optimization on assignments."""
    nv = len(G)
    embd = np.array([[1 / n for i in range(n)] for x in range(nv)])
    print(f"Memory before aggregation: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
    emb = dirtemporalAggregation(embd, G, alpha)

    kmeans = DifferentiableKMeans(n_clusters=n, beta=beta, max_iter=kmeans_iter)

    for i in range(iter):
        print(f"Iteration {i}")
        scaler = MinMaxScaler()
        emb1 = scaler.fit_transform(emb)
        print(f"Memory after scaling: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
        emb_torch = torch.from_numpy(emb1).float()
        assignments, centroids = kmeans.fit(emb_torch)
        print(f"Memory after K-means: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

        val = 1 - F.cosine_similarity(emb_torch.unsqueeze(1), centroids.unsqueeze(0), dim=2)
        val = val.detach().numpy()
        M = val.max(axis=1)
        m = val.min(axis=1)
        subx = (M.reshape(nv, 1) - val) / (M - m + 1e-10).reshape(nv, 1)
        su = subx.sum(axis=1)
        subx = subx / su.reshape(nv, 1)

        emb = laplacian_optimization(subx, G, assignments.numpy(), centroids.numpy(), alpha, lambda_weight, mu_weight)
        print(f"Memory after Laplacian: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
        emb = dirtemporalAggregation(emb, G, alpha)
        print(f"Memory after aggregation: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
    return emb

def dirtemporalSirGNStop(G, n, alpha, iter=100, beta=10.0, kmeans_iter=10, lambda_weight=1.0, mu_weight=1.0):
    """Run SIRGN with early stopping and Laplacian optimization."""
    nv = len(G)
    embd = np.array([[1 / n for i in range(n)] for x in range(nv)])
    print(f"Memory before aggregation: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
    emb = dirtemporalAggregation(embd, G, alpha)
    count = getnumber(emb)
    print('count', count)

    kmeans = DifferentiableKMeans(n_clusters=n, beta=beta, max_iter=kmeans_iter)

    for i in range(iter):
        print(f"Iteration {i}")
        scaler = MinMaxScaler()
        emb1 = scaler.fit_transform(emb)
        print(f"Memory after scaling: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
        emb_torch = torch.from_numpy(emb1).float()
        assignments, centroids = kmeans.fit(emb_torch)
        print(f"Memory after K-means: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

        val = 1 - F.cosine_similarity(emb_torch.unsqueeze(1), centroids.unsqueeze(0), dim=2)
        val = val.detach().numpy()
        M = val.max(axis=1)
        m = val.min(axis=1)
        subx = (M.reshape(nv, 1) - val) / (M - m + 1e-10).reshape(nv, 1)
        su = subx.sum(axis=1)
        subx = subx / su.reshape(nv, 1)

        emb2 = laplacian_optimization(subx, G, assignments.numpy(), centroids.numpy(), alpha, lambda_weight, mu_weight)
        print(f"Memory after Laplacian: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
        emb2 = dirtemporalAggregation(emb2, G, alpha)
        print(f"Memory after aggregation: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

        count1 = getnumber(emb2)
        print('count', count1)

        if count >= count1:
            print(f"Converged at iteration {i}, unique embeddings: {count1}")
            break
        else:
            emb = emb2
            count = count1
    return emb

def main(args):
    data = pd.read_csv(args.input)
    l = directed_loader.directed_loader()
    l.read(data)
    nv = len(l.G)


    edge_count = 0
    for v in range(nv):
        for t, lii, lio in l.G[v]:
            lii_list = lii.tolist() if lii is not None and lii.size > 0 else []
            lio_list = lio.tolist() if lio is not None and lio.size > 0 else []
            neighbors = lii_list + lio_list
            edge_count += len([(t, u) for u in neighbors])
    edge_count //= 2

    if args.stop:
        emb = dirtemporalSirGNStop(l.G, args.clusters, args.alpha, args.depth, args.beta, args.kmeans_iter, args.lambda_weight, args.mu_weight)
    else:
        emb = dirtemporalSirGN(l.G, args.clusters, args.alpha, args.depth, args.beta, args.kmeans_iter, args.lambda_weight, args.mu_weight)

    l.storeEmb(args.output, emb)

if __name__ == "__main__":
    args = parse_args()
    main(args)