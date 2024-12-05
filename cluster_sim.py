import argparse
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import scipy.stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import shutil


def extract_upper_triangle(matrix):
    matrix = np.array(matrix)
    rows, cols = matrix.shape

    extracted_matrix = []

    for i in range(rows):
        line = []
        for j in range(cols):
            if i != j:
                line.append(matrix[i, j])
        extracted_matrix.append(line)

    upper_triangle_elements = []

    for i in range(rows):
        for j in range(i + 1, cols):
            upper_triangle_elements.append(matrix[i, j])

    return upper_triangle_elements


def cos_sim_matrix(matrix):
    d = matrix @ matrix.T
    norm = (matrix * matrix).sum(axis=1, keepdims=True) ** .5
    return d / norm / norm.T


def main():
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--input_path', type=str,
                        help='Path to the input file')
    parser.add_argument('--input_vis_path', type=str,
                        help='Path to the input visualization directory')
    parser.add_argument('--target_dict_path', type=str,
                        help='Path to the target dict file')
    parser.add_argument('--target_numpy', type=str,
                        help='Path to the target numpy matrix')
    parser.add_argument('--output_path', type=str,
                        help='Path to the output file')
    parser.add_argument('--distance_threshold', type=float,
                        help='num of distance_threshold')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    with open(args.input_path, "rb") as file:
        id2word_all = pickle.load(file)

    with open(args.target_dict_path, "rb") as file:
        id2word_target = pickle.load(file)

    id2word_target_ = {v: k for k, v in id2word_target.items()}
    target_matrix = np.load(args.target_numpy)
    target_matrix_num = len(target_matrix)
    one_year_dict_num = len(id2word_all)
    slice = target_matrix_num // one_year_dict_num

    plt.rcParams["font.size"] = 20
    plt.tight_layout()
    time0_emb = []
    values = []

    emb_coses = []

    for key, value in tqdm(id2word_target.items()):
        emb = []
        values.append(value)
        for i in range(slice):
            emb.append(target_matrix[key+i*one_year_dict_num])
        emb_cos = cos_sim_matrix(np.array(emb))
        emb_cos = extract_upper_triangle(emb_cos)
        emb_cos = scipy.stats.zscore(emb_cos)
        emb_coses.append(emb_cos)

    emb_coses = np.array(emb_coses)
    agg_clustering = AgglomerativeClustering(n_clusters=None,
                                             distance_threshold=args.distance_threshold, linkage='ward')
    labels = agg_clustering.fit_predict(emb_coses)

    Z = linkage(emb_coses, method="ward", metric="euclidean")
    dendrogram(Z)
    plt.savefig(args.output_path + "dendrogram.png")

    word_cluster = {}

    for i, label in enumerate(labels):
        word_cluster[values[i]] = label

    grouped_data = defaultdict(list)

    for key, value in word_cluster.items():
        grouped_data[value].append(key)

    file = args.output_path + "cluster_result.txt"
    with open(file, 'w') as f:
        for value in sorted(grouped_data.keys()):
            print(f"Value: {value}", file=f)
            for key in grouped_data[value]:
                print(f"  {key}", file=f)

    vis_file = args.input_vis_path
    file = args.output_path

    for value in sorted(grouped_data.keys()):
        os.makedirs(file + str(value), exist_ok=True)
        for key in grouped_data[value]:
            trg_num = id2word_target_[key]
            trg_num_1 = trg_num // 100 * 100
            trg_num_2 = trg_num // 10 * 10
            shutil.copy(
                vis_file + f"/{trg_num_1}/{trg_num_2}/{trg_num}.png", file + str(value))


if __name__ == '__main__':
    main()
