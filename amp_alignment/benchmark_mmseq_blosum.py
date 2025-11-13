import time
import pandas as pd
from mean_mmseqs_score import compute_mean_normalized_bitscore


# PATHS_FOR_BENCHMARK = {
#     "positives": "../data/amp_data/amp_positives_MAX40.fasta",
#     "positives_hq": "../data/amp_data/amp_positives_hq_MAX40.fasta",
#     "negatives": "../data/amp_data/amp_negatives_MAX40.fasta",
#     "negatives_hq": "../data/amp_data/amp_negatives_hq_MAX40.fasta",
#     "random_uniform": "../data/amp_data/random_amp_uniform_distribution_MAX40.fasta",
#     "random_standard": "../data/amp_data/random_amp_with_standard_distribution_MAX40.fasta",
#     "UniProt": "../data/amp_data/uniprot_8_50_100_50K_MAX40.fasta",
#     "AMP-Diffusion": "../data/amp_data/amp-diffusion_MAX40.fasta",
#     "AMP-GAN": "../data/amp_data/amp-gan_MAX40_50K.fasta",
#     "CPL-Diff": "../data/amp_data/cpl-diff_MAX40.fasta",
#     "HydrAMP": "../data/amp_data/hydramp_MAX40.fasta",
#     "OmegAMP": "../data/amp_data/omegamp_MAX40_50K.fasta",
#     "AMP-LM": "../data/amp_data/amp_lm_MAX40.fasta",
#     "AMP-Muller": "../data/amp_data/amp_muller_MAX40.fasta",
# }

MAX_LEN = 40  # 25
MIN_LEN = 10  # 8

database = f"../data/similarity_control/sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}_vol_1.fasta"

PATHS_FOR_BENCHMARK = [
    f"../data/similarity_control/sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}_vol_1.fasta",
    f"../data/similarity_control/sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}_vol_2.fasta",
    f"../data/similarity_control/addition_or_deletion_begin_end_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}.fasta",
    f"../data/similarity_control/addition_or_deletion_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}.fasta",
    f"../data/similarity_control/mutated_1_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}.fasta",
    f"../data/similarity_control/mutated_2_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}.fasta",
    f"../data/similarity_control/mutated_3_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}.fasta",
    f"../data/similarity_control/mutated_5_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}.fasta",
    f"../data/similarity_control/mutated_7_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}.fasta",
    f"../data/similarity_control/shuffled_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}_vol_1.fasta",
    f"../data/similarity_control/random_2000_{MIN_LEN}_{MAX_LEN}_positives_hq_len_distrib.fasta",
    f"../data/similarity_control/reversed_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}_vol_1.fasta",
]


results = []
# for query in PATHS_FOR_BENCHMARK.values():
for query in PATHS_FOR_BENCHMARK:

    start_time = time.perf_counter()
    # database = PATHS_FOR_BENCHMARK["positives_hq"]
    mean_score, coverage = compute_mean_normalized_bitscore(query, database)
    elapsed_time = time.perf_counter() - start_time

    results.append(
        {
            "query": query.split("/")[-1],
            "database": database.split("/")[-1],
            "time_sec": elapsed_time,
            "mean_bitscore": mean_score,
            "coverage": coverage,
        }
    )
    print("Finished:", query, database)

df = pd.DataFrame(results)
df.to_csv("benchmark_results_mmseqs.csv", index=False)
