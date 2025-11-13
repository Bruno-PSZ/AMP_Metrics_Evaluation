import itertools
from multiprocessing import cpu_count, Pool, current_process
from typing import List
from Levenshtein import distance as levenshtein
from Bio import SeqIO
import numpy as np
from utils_sim import (
    common_subsequences,
    remove_first_occurrence,
    run_cdhit,
    get_clusters,
    get_cluster_representatives,
    get_representative_sequences,
)


def similarity(xa: List, xb: List) -> float:
    if len(xa) > len(xb):
        xa, xb = xb, xa
    na, nb = len(xa), len(xb)

    # Precompute membership by converting xb to a set, so that checking "in" is O(1)
    xb_set = set(xb)
    s1 = sum(1 for x in xa if x in xb_set) / (2 * na)

    # Obtain common subsequences (sorted descending by length)
    common = common_subsequences(xa, xb)
    xa_copy, xb_copy = xa[:], xb[:]
    s2 = 0

    for c in common:
        new_xa = remove_first_occurrence(xa_copy, c)
        new_xb = remove_first_occurrence(xb_copy, c)
        # Ensure that a removal took place in both sequences.
        if len(new_xa) < len(xa_copy) and len(new_xb) < len(xb_copy):
            xa_copy = new_xa
            xb_copy = new_xb
            s2 += 2 * len(c)

    s2 = s2 / (2 * na + 2 * nb)
    return s1 + s2


def process_pair(pair):
    xa, xb = pair
    return similarity(xa, xb)


def calculate_similarity(sequences, n=1000, replace=False, processes=None):
    if n > len(sequences):
        replace = True
    seqs = np.random.choice(sequences, n, replace=replace)
    sequence_pairs = itertools.combinations(seqs, 2)
    if processes is None:
        processes = cpu_count()  # Use all available CPUs

    with Pool(processes=processes) as pool:
        results = pool.map(process_pair, sequence_pairs)

    return results


def calculate_similarity_matrix(sequences_1: list, sequences_2: list) -> np.ndarray:
    processes = cpu_count()
    pairs = itertools.product(sequences_1, sequences_2)
    with Pool(processes=processes) as pool:
        results = pool.map(process_pair, pairs)

    results = np.array(results).reshape(len(sequences_1), len(sequences_2))
    return results


def mean_similarity(sequences_1: list, sequences_2: list) -> float:
    sim_matrix = calculate_similarity_matrix(sequences_1, sequences_2)
    row_mins = np.max(sim_matrix, axis=1)
    return np.mean(row_mins)


def mean_similarity_cd_hit(
    cd_hit_path,
    sequences_1_path,
    sequences_2_path,
    output_path_1,
    output_path_2,
    threshold,
    vocab_size=5,
    filter_small_clusters=True,
    verbose=True,
):
    run_cdhit(
        cd_hit_path, sequences_1_path, output_path_1, threshold, vocab_size, verbose
    )
    run_cdhit(
        cd_hit_path, sequences_2_path, output_path_2, threshold, vocab_size, verbose
    )

    clusters_1 = get_clusters(output_path_1, filter_small_clusters)
    clusters_2 = get_clusters(output_path_2, filter_small_clusters)

    representatives = get_cluster_representatives(clusters_1)
    representatives_2 = get_cluster_representatives(clusters_2)

    sequences_1 = get_representative_sequences(representatives, output_path_1)
    sequences_2 = get_representative_sequences(representatives_2, output_path_2)

    return mean_similarity(sequences_1, sequences_2)


def calculate_levenshtein(sequences, n=1000, replace=False):
    if n > len(sequences):
        replace = True
    seqs = np.random.choice(sequences, n, replace=replace)
    pairs = itertools.combinations(seqs, 2)
    results = []
    for a, b in pairs:
        dist = levenshtein(a, b)
        results.append(dist)
    return results


def levenshtein_distance_matrix(sequences_1: list, sequences_2: list) -> np.ndarray:
    result = np.zeros((len(sequences_1), len(sequences_2)))
    for i, seq1 in enumerate(sequences_1):
        for j, seq2 in enumerate(sequences_2):
            result[i, j] = levenshtein(seq1, seq2)
    return result


def levenshtein_similarity_matrix(sequences_1: list, sequences_2: list) -> np.ndarray:
    result = np.zeros((len(sequences_1), len(sequences_2)))
    for i, seq1 in enumerate(sequences_1):
        for j, seq2 in enumerate(sequences_2):
            result[i, j] = 1 - (levenshtein(seq1, seq2) / max(len(seq1), len(seq2)))
    return result


def mean_levenshtein_similarity(sequences_1: list, sequences_2: list) -> float:
    lev_matrix = levenshtein_similarity_matrix(sequences_1, sequences_2)
    row_mins = np.max(lev_matrix, axis=1)
    return np.mean(row_mins)


def mean_levenshtein_similarity_cd_hit(
    cd_hit_path,
    sequences_1_path,
    sequences_2_path,
    output_path_1,
    output_path_2,
    threshold,
    vocab_size=5,
    filter_small_clusters=True,
    verbose=True,
):
    run_cdhit(
        cd_hit_path, sequences_1_path, output_path_1, threshold, vocab_size, verbose
    )
    run_cdhit(
        cd_hit_path, sequences_2_path, output_path_2, threshold, vocab_size, verbose
    )

    clusters_1 = get_clusters(output_path_1, filter_small_clusters)
    clusters_2 = get_clusters(output_path_2, filter_small_clusters)

    representatives = get_cluster_representatives(clusters_1)
    representatives_2 = get_cluster_representatives(clusters_2)

    sequences_1 = get_representative_sequences(representatives, output_path_1)
    sequences_2 = get_representative_sequences(representatives_2, output_path_2)

    original_number_of_sequences = sum(
        1 for _ in SeqIO.parse(sequences_2_path, "fasta")
    )
    after_clustering_number_of_sequences = len(sequences_2)

    print(
        "Coverage:", after_clustering_number_of_sequences / original_number_of_sequences
    )

    return (
        mean_levenshtein_similarity(sequences_1, sequences_2),
        after_clustering_number_of_sequences / original_number_of_sequences,
    )
