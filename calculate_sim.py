import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from tqdm import tqdm


def cos_sim_matrix(matrix):
    d = matrix @ matrix.T
    norm = (matrix * matrix).sum(axis=1, keepdims=True) ** .5
    return d / norm / norm.T


def main():
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--input_path', type=str,
                        help='Path to the input file')
    parser.add_argument('--target_dict_path', type=str,
                        help='Path to the target dict file')
    parser.add_argument('--target_numpy', type=str,
                        help='Path to the target numpy matrix')
    parser.add_argument('--output_path', type=str,
                        help='Path to the output file')
    args = parser.parse_args()

    with open(args.input_path, "rb") as file:
        id2word_all = pickle.load(file)

    with open(args.target_dict_path, "rb") as file:
        id2word_target = pickle.load(file)

    target_matrix = np.load(args.target_numpy)
    target_matrix_num = len(target_matrix)
    one_year_dict_num = len(id2word_all)
    slice = target_matrix_num // one_year_dict_num

    plt.rcParams["font.size"] = 20
    plt.tight_layout()

    for key, value in tqdm(id2word_target.items()):
        emb = []
        for i in range(slice):
            emb.append(target_matrix[key+i*one_year_dict_num])

        emb_cos = cos_sim_matrix(np.array(emb))
        plt.figure()
        sns.heatmap(emb_cos, cmap="jet")
        plt.title(value)
        plt.xlabel("period")
        plt.ylabel("period")

        plt.tight_layout()

        base = args.output_path
        os.makedirs(base, exist_ok=True)
        os.makedirs(base+"/{}".format(key // 100 * 100), exist_ok=True)
        os.makedirs(base + "/{}/{}".format(key // 100 *
                    100, key // 10 * 10), exist_ok=True)
        plt.savefig(base + '/{}/{}/{}.png'.format(key //
                    100 * 100, key // 10 * 10, key))
        plt.clf()
        plt.close()


if __name__ == '__main__':
    main()
