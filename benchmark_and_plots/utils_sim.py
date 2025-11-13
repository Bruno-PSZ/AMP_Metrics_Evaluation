from Bio import SeqIO
import matplotlib.pyplot as plt
import random
from typing import List, Literal
from matplotlib.axes import Axes
import numpy as np
import subprocess
from itertools import groupby
import shlex
import time

def plot_hist(
    data: np.ndarray,
    xlabel: str,
    color: str = "#68d6bc",
    ytype: Literal["frequency", "density"] = "density",
    figsize: tuple[int, int] = (4, 3),
    bins: int = 10,
    alpha: float = 1.0,
    edgecolor: str = "black",
    linewidth: float = 1.0,
    label: str | None = None,
    ax: Axes | None = None,
):
    arr = np.asarray(data).ravel()
    if arr.size == 0:
        raise ValueError("Input array is empty.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    ylabels = {"density": "Density", "frequency": "Frequency"}

    ax.hist(
        arr,
        bins=bins,
        color=color,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        density=(ytype == "density"),
        label=label,
    )

    if label is not None:
        ax.legend()

    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabels[ytype])

    if created_fig:
        plt.tight_layout()
        plt.show()


def shuffle_sequences(sequences: list[str], seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    shuffled = []
    for seq in sequences:
        chars = list(seq)
        rng.shuffle(chars)
        shuffled.append("".join(chars))
    return shuffled


def reverse_sequences(sequences: list[str]) -> list[str]:
    return [seq[::-1] for seq in sequences]


def random_subset(sequences: list[str], n_samples: int, seed: int = 42) -> list[str]:
    if n_samples > len(sequences):
        raise ValueError(f"Cannot sample {n_samples} sequences from a list of length {len(sequences)}.")

    rng = random.Random(seed)
    return rng.sample(sequences, n_samples)


def read_fasta_file(path: str) -> list[str]:
    sequences: list[str] = []
    current_seq: list[str] = []

    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            if line.startswith(">"):
                if current_seq:
                    sequence = "".join(current_seq)
                    if sequence:
                        sequences.append(sequence)
                    current_seq = []
            else:
                current_seq.append(line)

        # Add the last sequence if present
        if current_seq:
            sequence = "".join(current_seq)
            if sequence:
                sequences.append(sequence)

    return sequences


def write_to_fasta_file(
    sequences: list[str],
    path: str,
    headers: list[str] | None = None,
):
    with open(path, "w") as f:
        for i, seq in enumerate(sequences):
            header = headers[i] if headers else f">sequence_{i + 1}"
            f.write(f"{header}\n")
            f.write(f"{seq}\n")

def get_sequences(file):
    seqs = []
    for record in SeqIO.parse(file, 'fasta'):
        seqs.append(str(record.seq))
    return seqs


def run_cdhit(cd_hit_path, input_path, output_path, threshold, vocab_size=5, verbose=True):
    """Run CD-HIT with the given parameters."""
    cmd = f"{shlex.quote(cd_hit_path)} -i {shlex.quote(input_path)} -o {shlex.quote(output_path)} -c {threshold} -n {vocab_size}"
    
    if verbose:
        # Show CD-HIT output in console
        subprocess.run(cmd, shell=True, check=True)
    else:
        # Suppress CD-HIT output
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        
def get_clusters(output_path, filter_small_clusters=True):
    """Parse CD-HIT cluster output and return clusters."""
    output_cluster_path = output_path + '.clstr'
    
    clusters = []
    with open(output_cluster_path, 'r') as f:
        for _, group in groupby(f, lambda line: line.startswith('>')):
            cluster = list(group)
            if not cluster[0].startswith('>'):  # Ignore headers
                clusters.append(cluster)

    # Filter clusters smaller than 2 sequences
    return [c for c in clusters if len(c) > 1] if filter_small_clusters else clusters


def get_cluster_representatives(clusters):
    """Extract representative sequences from clusters."""
    return [
        seq.split('>')[1].split('...')[0]
        for cluster in clusters
        for seq in cluster if '*' in seq
    ]


def get_representative_sequences(representatives, output_path):
    """Retrieve representative sequences from the FASTA file."""
    bio_sequences = {record.id: str(record.seq) for record in SeqIO.parse(output_path, 'fasta')}
    return [bio_sequences.get(rep, f"Missing: {rep}") for rep in representatives]


def common_subsequences(seq1: List, seq2: List) -> List[List]:
    len1, len2 = len(seq1), len(seq2)

    # Use two rows for DP to save memory
    dp_prev = [0] * (len2 + 1)
    matches = []

    # Build DP table for longest common suffix lengths,
    # collecting matches of length greater than 1.
    for i in range(1, len1 + 1):
        dp_curr = [0] * (len2 + 1)
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp_curr[j] = dp_prev[j - 1] + 1
                if dp_curr[j] > 1:
                    # Record the start positions and the common subsequence length.
                    matches.append((i - dp_curr[j], j - dp_curr[j], dp_curr[j]))
            else:
                dp_curr[j] = 0
        dp_prev = dp_curr

    # Sort matches by descending length.
    matches.sort(key=lambda x: -x[2])

    # Use lists of booleans for used positions (faster than set intersections)
    accepted = []
    used1 = [False] * len1
    used2 = [False] * len2

    for start1, start2, length in matches:
        # Check if any positions in this subsequence have been used
        if any(used1[i] for i in range(start1, start1 + length)) or any(
                used2[j] for j in range(start2, start2 + length)):
            continue
        accepted.append(seq1[start1:start1 + length])
        for i in range(start1, start1 + length):
            used1[i] = True
        for j in range(start2, start2 + length):
            used2[j] = True

    return accepted


def remove_first_occurrence(seq: List, subseq: List) -> List:
    n, m = len(seq), len(subseq)
    # Iterate until we find a matching slice.
    for i in range(n - m + 1):
        if seq[i:i + m] == subseq:
            # Remove the first occurrence and return the new sequence.
            return seq[:i] + seq[i + m:]
    return seq


def time_function(function, *args, **kwargs) -> tuple:
    start_time = time.perf_counter()
    result = function(*args, **kwargs)
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    print(f"Execution time: {time_taken:.4f} seconds")
    return (result, time_taken)