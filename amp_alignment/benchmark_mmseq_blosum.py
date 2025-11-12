import time
import pandas as pd
from mean_mmseqs_score import compute_mean_normalized_bitscore


# PATHS_FOR_BENCHMARK = {
#     "positives": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/amp_positives_MAX40.fasta",
#     "positives_hq": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/amp_positives_hq_MAX40.fasta",
#     "negatives": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/amp_negatives_MAX40.fasta",
#     "negatives_hq": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/amp_negatives_hq_MAX40.fasta",
#     "random_uniform": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/random_amp_uniform_distribution_MAX40.fasta",
#     "random_standard": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/random_amp_with_standard_distribution_MAX40.fasta",
#     "UniProt": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/uniprot_8_50_100_50K_MAX40.fasta",
#     "AMP-Diffusion": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/amp-diffusion_MAX40.fasta",
#     "AMP-GAN": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/amp-gan_MAX40_50K.fasta",
#     "CPL-Diff": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/cpl-diff_MAX40.fasta",
#     "HydrAMP": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/hydramp_MAX40.fasta",
#     "OmegAMP": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/omegamp_MAX40_50K.fasta",
#     "AMP-LM": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/amp_lm_MAX40.fasta",
#     "AMP-Muller": "/raid/brunopsz/Metrics_Eval/FINAL_MAX_40AA_AMP/amp_muller_MAX40.fasta",
# }

MAX_LEN = 40  # 25
MIN_LEN = 10  # 8

database = f"/raid/brunopsz/PLAST_EXPERIMENTS/data_{MAX_LEN}/sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}_vol_1.fasta"

PATHS_FOR_BENCHMARK = [
    f"/raid/brunopsz/PLAST_EXPERIMENTS/data_{MAX_LEN}/sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}_vol_1.fasta",
    f"/raid/brunopsz/PLAST_EXPERIMENTS/data_{MAX_LEN}/sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}_vol_2.fasta",
    f"/raid/brunopsz/PLAST_EXPERIMENTS/data_{MAX_LEN}/addition_or_deletion_begin_end_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}.fasta",
    f"/raid/brunopsz/PLAST_EXPERIMENTS/data_{MAX_LEN}/addition_or_deletion_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}.fasta",
    f"/raid/brunopsz/PLAST_EXPERIMENTS/data_{MAX_LEN}/mutated_1_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}.fasta",
    f"/raid/brunopsz/PLAST_EXPERIMENTS/data_{MAX_LEN}/mutated_2_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}.fasta",
    f"/raid/brunopsz/PLAST_EXPERIMENTS/data_{MAX_LEN}/mutated_3_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}.fasta",
    f"/raid/brunopsz/PLAST_EXPERIMENTS/data_{MAX_LEN}/mutated_5_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}.fasta",
    f"/raid/brunopsz/PLAST_EXPERIMENTS/data_{MAX_LEN}/mutated_7_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}.fasta",
    f"/raid/brunopsz/PLAST_EXPERIMENTS/data_{MAX_LEN}/shuffled_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}_vol_1.fasta",
    f"/raid/brunopsz/PLAST_EXPERIMENTS/data_{MAX_LEN}/random_2000_{MIN_LEN}_{MAX_LEN}_positives_hq_len_distrib.fasta",
    f"/raid/brunopsz/PLAST_EXPERIMENTS/data_{MAX_LEN}/reversed_sampled_positives_hq_1000_{MIN_LEN}_{MAX_LEN}_vol_1.fasta",
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
