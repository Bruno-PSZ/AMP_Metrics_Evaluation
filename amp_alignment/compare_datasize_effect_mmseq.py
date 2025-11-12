import time
import pandas as pd
from mean_mmseqs_score import compute_mean_normalized_bitscore
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import tempfile
from pathlib import Path


def random_subset(sequences: list[str], n_samples: int, seed: int = 42) -> list[str]:
    if n_samples > len(sequences):
        raise ValueError(
            f"Cannot sample {n_samples} sequences from a list of length {len(sequences)}."
        )

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


def write_fasta(sequences, filepath):
    """Write a list of sequences to a FASTA file."""
    with open(filepath, "w") as f:
        for i, seq in enumerate(sequences, 1):
            f.write(f">seq{i}\n{seq}\n")


def time_function(function, timeout=180, *args, **kwargs) -> tuple:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(function, *args, **kwargs)
        try:
            start_time = time.perf_counter()
            result = future.result(timeout=timeout)
            end_time = time.perf_counter()
            time_taken = end_time - start_time
            print(f"Execution time: {time_taken:.4f} seconds")
            return (result, time_taken)
        except TimeoutError:
            print(f"Execution exceeded {timeout}s, returning defaults.")
            return (0, timeout + 0.001)  # store just over 60s as marker


results = []
N_SAMPLES = [1000, 2000, 3000, 5000, 7000, 10000, 14941]

PATHS = {
    "positives": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/amp_positives_MAX40.fasta",
    "OmegAMP": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/omegamp_MAX40.fasta",
}

datasets = {name: read_fasta_file(path) for name, path in PATHS.items()}

for n_samples in N_SAMPLES:
    print(f"Running for n_samples={n_samples}")

    reference = random_subset(datasets["positives"], n_samples)
    model = random_subset(datasets["OmegAMP"], n_samples)

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = Path(tmpdir) / "reference.fasta"
        model_path = Path(tmpdir) / "model.fasta"

        write_fasta(reference, ref_path)
        write_fasta(model, model_path)
        value, t = time_function(
            compute_mean_normalized_bitscore, 180, str(model_path), str(ref_path)
        )

    results.append({"n_samples": n_samples, "value": value, "time": t})

df = pd.DataFrame(results)
df.to_csv("compare_datasize_effect_mmseqs.csv", index=False)
print(df)