# Analyzing Continuous Semantic Shifts with Diachronic Word Similarity Matrices
This is the repository for our paper "Analyzing Continuous Semantic Shifts with Diachronic Word Similarity Matrices."

## Abstract
The meanings and relationships of words shift over time. 
This phenomenon is referred to as semantic shift.
Research focused on understanding how semantic shifts occur over multiple time periods is essential for gaining a detailed understanding of semantic shifts.
However, detecting change points only between adjacent time periods is insufficient for analyzing detailed semantic shifts, and using BERT-based methods to examine word sense proportions incurs a high computational cost.
To address those issues, we propose a simple yet intuitive framework for how semantic shifts occur over multiple time periods by utilizing similarity matrices based on word embeddings.
We calculate diachronic word similarity matrices using fast and lightweight word embeddings across arbitrary time periods, making it deeper to analyze continuous semantic shifts.
Additionally, by clustering the resulting similarity matrices, we can categorize words that exhibit similar behavior of semantic shift in an unsupervised manner.

- Data:
    - [COHA](https://www.english-corpora.org/coha/)
    - [COCA](https://www.english-corpora.org/coca/)
    - [Mainichi Shimbun](https://mainichi.jp/contents/edu/03.html)
    - [pseudo](https://github.com/alan-turing-institute/room2glo)
- Model:
    - [SPPMI-SVD joint](https://github.com/a1da4/pmi-semantic-difference/tree/main/models/pmi-svd)
- Preprocess:
    - [CCOHA](https://github.com/Reem-Alatrash/CCOHA)
- Source Code:
    - `calculate_sim.py`
    - `cluster_sim.py`

## 1\. Setup
### From requirements.txt
Python `>= 3.8`
```
pip install -r requirements.txt
```

## 2\. Preprocessing and Learning word embedding
When using this method, word embeddings aligned for each time period are required as input.
Any method can be used, but in this study, we employ SPPMI-SVD joint to ensure fast and lightweight processing.

## 3\. Visualization of Similarity Matrices
In this script, similarity matrices are calculated from word embeddings for each time period and visualized.
--target_numpy is expected to be a numpy file containing concatenated word embeddings for each time period.
As output, the script provides a visualization of the diachronic word similarity matrix.

```
python3 ./calculate_sim.py \
    --input_path {PATH_TO_INPUT_DICT_FILE} \
    --target_dict_path {PATH_TO_TARGET_DICT_FILE} \
    --target_numpy {PATH_TO_WORD_EMBEDDINGS}\
    --output_path {PATH_TO_OUTPUT_FILE} 
```

## 4\. Clustering of Similarity Matrices
This script performs hierarchical clustering using the standardized upper triangular components of the similarity matrix as features.
As output, the script provides a file showing the results of hierarchical clustering and a visualization of the similarity matrices for each cluster.

```
python ./cluster_sim.py \
    --input_path {PATH_TO_INPUT_DICT_FILE} \
    --input_vis_path {PATH_TO_INPUT_VIS_FILE} \
    --target_dict_path {PATH_TO_TARGET_DICT_FILE} \
    --target_numpy {PATH_TO_WORD_EMBEDDINGS} \
    --output_path {PATH_TO_OUTPUT_FILE} \
    --distance_threshold {NUM_OF_THRESHOLD}
```
