import subprocess
import pandas as pd
import os
import argparse

from Bio import SeqIO


MMSEQ_BIN = "./MMseqs2/build/bin/mmseqs"
PROFILE = "easy-search"


def get_num_sequences(fasta_path):
    return sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))


def safe_load_tsv(path):
    if os.path.getsize(path) == 0:
        return pd.DataFrame(columns=[0, 11])  # empty DF with required cols
    return pd.read_csv(path, sep="\t", header=None)


def run_mmseq(query_fasta, database, i, add_self_matches=False):
    cmd = [
        MMSEQ_BIN,
        PROFILE,
        query_fasta,
        database,
        f"result_{i}.tsv",
        "tmp",
        "-v",
        "0",
        "-e",
        "1000",
    ]

    if add_self_matches:
        cmd += ["--add-self-matches", "--max-seqs", "1"]

    subprocess.run(cmd, check=True)


def compute_mean_normalized_bitscore(
    query_fasta: str, database_fasta: str, output_results_csv: str | None = None
) -> tuple[float, float]:
    i = 0

    # --- Step 1: Load sequences ---
    query_records = list(SeqIO.parse(query_fasta, "fasta"))
    db_records = list(SeqIO.parse(database_fasta, "fasta"))
    db_seqs = {str(r.seq) for r in db_records}

    # --- Step 2: Separate shared vs unique query sequences ---
    shared_records = []
    filtered_query_records = []
    for r in query_records:
        if str(r.seq) in db_seqs:
            # print(str(r.seq))
            shared_records.append(r)
        else:
            filtered_query_records.append(r)
    shared_count = len(shared_records)

    # Save filtered queries to file
    filtered_query_fasta = query_fasta.split(".")[-1] + "_filtered.fasta"
    SeqIO.write(filtered_query_records, filtered_query_fasta, "fasta")

    # If no queries left after filtering → all shared with DB
    if not filtered_query_records:

        if output_results_csv is not None:
            df_shared = pd.DataFrame(
                {
                    "queryId": [r.id for r in shared_records],
                    "targetId": [None] * shared_count,
                    "bit_score": [None] * shared_count,
                    "self_targetId": [r.id for r in shared_records],
                    "self_bit_score": [None] * shared_count,
                    "normalized": [1.0] * shared_count,
                }
            )
            df_shared.to_csv(output_results_csv, index=False)

        print(
            f"All queries are shared with the database for query {query_fasta.split('/')[-1]}. Returning 1.0"
        )
        return 1.0, 1.0

    # --- Step 3: Run filtered query vs database ---
    run_mmseq(filtered_query_fasta, database_fasta, i, add_self_matches=False)
    df1 = safe_load_tsv(f"result_{i}.tsv")

    if df1.empty:
        print("No hits found in query vs database.")
        total_queries = len(query_records)
        mean_score = shared_count / total_queries
        return mean_score, mean_score

    idx = df1.groupby(0)[df1.columns[-1]].idxmax()
    result_1 = df1.loc[idx, [0, 1, df1.columns[-1]]]
    result_1.columns = ["queryId", "targetId", "bit_score"]
    i += 1

    # --- Step 4: Run self-alignment for normalization ---
    run_mmseq(filtered_query_fasta, filtered_query_fasta, i, add_self_matches=True)
    df2 = safe_load_tsv(f"result_{i}.tsv")

    if df2.empty:
        print("No self-hits found.")
        total_queries = len(query_records)
        mean_score = shared_count / total_queries
        return mean_score, mean_score

    idx = df2.groupby(0)[df2.columns[-1]].idxmax()
    result_2 = df2.loc[idx, [0, 1, df2.columns[-1]]]
    result_2.columns = ["queryId", "self_targetId", "self_bit_score"]

    # --- Step 5: Normalize scores ---
    merged = pd.merge(result_1, result_2, on="queryId", how="inner")
    merged["normalized"] = merged["bit_score"] / merged["self_bit_score"]

    total_queries = len(query_records)
    queries_with_hits = set(merged["queryId"])
    num_no_hits = len(filtered_query_records) - len(queries_with_hits)

    if shared_records:
        shared_df = pd.DataFrame(
            {
                "queryId": [r.id for r in shared_records],
                "targetId": [r.id for r in shared_records],
                "bit_score": [None] * shared_count,
                "self_targetId": [r.id for r in shared_records],
                "self_bit_score": [None] * shared_count,
                "normalized": [1.0] * shared_count,
            }
        )
        merged = pd.concat([merged, shared_df], ignore_index=True)

    if output_results_csv is not None:
        merged.to_csv(output_results_csv, index=False)

    if any(merged["normalized"] > 1):
        print("Warning: Some normalized bitscores > 1")
        print(merged[merged["normalized"] > 1])

    # --- Step 6: Account for shared + no-hits ---

    all_scores = list(merged["normalized"]) + [0] * num_no_hits + [1] * shared_count
    mean_norm = pd.Series(all_scores).mean()

    print(f"Total queries: {total_queries}")
    print(f"Shared with DB (count as 1): {shared_count}")
    print(f"No-hits (count as 0): {num_no_hits}")
    print(f"Queries with hits: {len(queries_with_hits)}")
    print(f"Final mean normalized bitscore: {mean_norm}")

    coverage = (len(queries_with_hits) + shared_count) / total_queries

    return mean_norm, coverage


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute mean normalized bitscore using MMSEQ2."
    )
    parser.add_argument("--query", required=True, help="Query FASTA file")
    parser.add_argument("--database", required=True, help="Database FASTA file")

    args = parser.parse_args()

    mean_score = compute_mean_normalized_bitscore(args.query, args.database)
    print("Mean normalized bitscore:", mean_score)
