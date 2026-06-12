"""
AFR NMF Pipeline v2 — Fast Biomarker Subset
=============================================
Rewritten after discovering:
  1. Pan-UKBB per-phenotype files have NO `rsid` column — must match
     on (chr, pos, ref, alt) against allvar.pruned.closesttss.hugo
  2. Per-phenotype files have NO sample-size columns (n_cases_AFR etc.)
     — these come from the separate full phenotype manifest
     (panukbb_phenotype_manifest_tsv.bgz), joined on
     [phenocode, pheno_sex, trait_type, coding, modifier]
  3. Files are ~1.5GB each (genome-wide, all ancestries) — too slow to
     process all 1,493 AFR-eligible phenotypes in a few hours.

This version runs on a SMALL SUBSET (default: 15 biomarker phenotypes
with the highest AFR Neff, ~6,200 each) for a feasible ~1.5-2hr runtime,
producing the same outputs (Z-score matrix, NMF, EUR comparison).

REQUIRED FILES in 'data copy/':
  trait_manifest_TableS6.xlsx
  allvar.pruned.closesttss.hugo
  FactorGo.Zm.tsv.gz
  tSVD.U.tsv.gz
  pheno_manifest_full.tsv.bgz   <-- NEW: rename your
                                     panukbb_phenotype_manifest_tsv.bgz to this
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import requests
import time
import gzip
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── CONFIGURATION ─────────────────────────────────────────────────────────
DATA_DIR = 'data copy'
RESULTS_DIR = 'results copy'
DOWNLOAD_DIR = 'afr_downloads'
K_NMF = 10                  # fewer components since n_phenotypes is small (15)
N_SUBSET = 15               # number of AFR phenotypes to run (by highest Neff)
TRAIT_TYPE_FILTER = 'biomarkers'  # restrict subset to this trait_type (fast, high-Neff)
MIN_AFR_N_EFF = 100
DELETE_AFTER_EXTRACT = True

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. LOAD REFERENCE DATA ────────────────────────────────────────────────
print("=" * 65)
print("STEP 1: Loading Reference Data")
print("=" * 65)

manifest = pd.read_excel(f'{DATA_DIR}/trait_manifest_TableS6.xlsx', header=1)
manifest = manifest.reset_index(drop=True)
manifest['phenocode'] = manifest['phenocode'].astype(str)

hugo = pd.read_csv(f'{DATA_DIR}/allvar.pruned.closesttss.hugo', sep='\t')
target_snp_list = hugo['rsid'].values
n_snps = len(target_snp_list)

# Build (chr, pos, ref, alt) -> rsid lookup. Pan-UKBB files use chr as string.
hugo['chr_str'] = hugo['chr'].astype(str)
snp_key_to_rsid = {}
snp_key_to_col = {}
for i, r in hugo.iterrows():
    key = (r['chr_str'], int(r['pos']), r['ref'], r['alt'])
    snp_key_to_rsid[key] = r['rsid']
    snp_key_to_col[key] = i

Zm_eur = pd.read_csv(f'{DATA_DIR}/FactorGo.Zm.tsv.gz', sep='\t', header=None)
Zm_eur.columns = [f'FG_F{i+1}' for i in range(100)]

print(f"  Manifest:        {manifest.shape[0]} phenotypes")
print(f"  Target SNPs:     {n_snps:,}")
print(f"  EUR Zm (FactorGo): {Zm_eur.shape}")

# ── 2. JOIN AFR SAMPLE SIZES & SELECT SUBSET ────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Joining AFR Sample Sizes & Selecting Subset")
print("=" * 65)

afr_full = pd.read_csv(f'{DATA_DIR}/pheno_manifest_full.tsv.bgz', sep='\t',
                       compression='gzip')
afr_full['phenocode'] = afr_full['phenocode'].astype(str)

merge_cols = ['phenocode', 'pheno_sex', 'trait_type', 'coding', 'modifier']
merged = manifest.merge(
    afr_full[merge_cols + ['n_cases_AFR', 'n_controls_AFR']],
    on=merge_cols, how='left'
)
merged['n_eff_AFR'] = np.where(
    merged['n_controls_AFR'].notna(),
    2 / (1/merged['n_cases_AFR'] + 1/merged['n_controls_AFR']),
    merged['n_cases_AFR']
)

eligible = merged[merged['n_eff_AFR'] >= MIN_AFR_N_EFF].copy()
if TRAIT_TYPE_FILTER:
    eligible = eligible[eligible['trait_type'] == TRAIT_TYPE_FILTER]

eligible = eligible.sort_values('n_eff_AFR', ascending=False).head(N_SUBSET)
eligible = eligible.reset_index(drop=True)

print(f"  AFR-eligible phenotypes (Neff>={MIN_AFR_N_EFF}): {len(merged[merged['n_eff_AFR']>=MIN_AFR_N_EFF])}")
print(f"  Trait type filter: {TRAIT_TYPE_FILTER}")
print(f"  Selected subset:   {len(eligible)} phenotypes")
print(eligible[['phenocode', 'description', 'n_eff_AFR']].to_string(index=False))

eligible.to_csv(f'{RESULTS_DIR}/afr_subset_selected.csv', index=False)

# ── 3. DOWNLOAD & EXTRACT ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Downloading AFR GWAS Summary Statistics")
print("=" * 65)
print("""
  Matching variants by (chr, pos, ref, alt) since per-phenotype files
  have no rsid column. Extracting beta_AFR, se_AFR.
""")


def download_and_extract(aws_link, filename, snp_key_to_col, download_dir,
                          delete_after=True, verbose=True):
    local_path = os.path.join(download_dir, filename)
    t0 = time.time()

    if not os.path.exists(local_path):
        try:
            response = requests.get(aws_link, timeout=600, stream=True)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        except Exception as e:
            if verbose:
                print(f"    [DOWNLOAD FAIL] {filename}: {e}")
            return None

    t_dl = time.time() - t0
    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    t1 = time.time()

    z_raw = {}  # col_idx -> beta_AFR/se_AFR (raw Z)
    n_found = 0

    # Pre-build lookup arrays for vectorized matching
    keys_arr = np.array(list(snp_key_to_col.keys()), dtype=object)
    key_set = set(snp_key_to_col.keys())

    try:
        with gzip.open(local_path, 'rt') as fh:
            for chunk in pd.read_csv(fh, sep='\t', chunksize=500000,
                                      usecols=lambda c: c in (
                                          'chr', 'pos', 'ref', 'alt',
                                          'beta_AFR', 'se_AFR'),
                                      low_memory=False):
                if not {'chr', 'pos', 'ref', 'alt', 'beta_AFR', 'se_AFR'}.issubset(chunk.columns):
                    if verbose:
                        print(f"    [MISSING COLS] {filename}: {chunk.columns.tolist()}")
                    break

                chunk = chunk.dropna(subset=['beta_AFR', 'se_AFR'])
                if chunk.empty:
                    continue
                chunk = chunk[chunk['se_AFR'] > 0]
                if chunk.empty:
                    continue

                # Vectorized key construction
                chunk_keys = list(zip(chunk['chr'].astype(str),
                                       chunk['pos'].astype(int),
                                       chunk['ref'],
                                       chunk['alt']))
                mask = [k in key_set for k in chunk_keys]
                if not any(mask):
                    continue

                sub = chunk[mask]
                sub_keys = [k for k, m in zip(chunk_keys, mask) if m]
                betas = sub['beta_AFR'].values
                ses = sub['se_AFR'].values

                for key, beta, se in zip(sub_keys, betas, ses):
                    col = snp_key_to_col[key]
                    z_raw[col] = beta / se
                    n_found += 1

    except Exception as e:
        if verbose:
            print(f"    [PARSE FAIL] {filename}: {e}")
        if delete_after and os.path.exists(local_path):
            os.remove(local_path)
        return None

    t_parse = time.time() - t1
    if verbose:
        print(f"    [{filename}] size={file_size_mb:.1f}MB "
              f"download={t_dl:.1f}s parse={t_parse:.1f}s "
              f"snps_found={n_found}/{len(snp_key_to_col)}")

    if delete_after and os.path.exists(local_path):
        os.remove(local_path)

    return {'z_raw': z_raw, 'n_found': n_found,
            'file_size_mb': file_size_mb, 'download_sec': t_dl,
            'parse_sec': t_parse}


results = []
for i, row in eligible.iterrows():
    print(f"\n  [{i+1}/{len(eligible)}] {row['description'][:50]} "
          f"(phenocode={row['phenocode']}, Neff_AFR={row['n_eff_AFR']:.0f})")
    res = download_and_extract(
        aws_link=row['aws_link'],
        filename=row['filename'],
        snp_key_to_col=snp_key_to_col,
        download_dir=DOWNLOAD_DIR,
        delete_after=DELETE_AFTER_EXTRACT,
        verbose=True
    )
    if res is None:
        continue
    res['phenocode'] = row['phenocode']
    res['description'] = row['description']
    res['n_eff_AFR'] = row['n_eff_AFR']
    results.append(res)

print(f"\n  Successfully processed: {len(results)}/{len(eligible)}")

if results:
    sizes = [r['file_size_mb'] for r in results]
    dls = [r['download_sec'] for r in results]
    parses = [r['parse_sec'] for r in results]
    print(f"\n  TIMING SUMMARY:")
    print(f"    Avg file size:  {np.mean(sizes):.1f} MB")
    print(f"    Avg download:   {np.mean(dls):.1f} sec")
    print(f"    Avg parse:      {np.mean(parses):.1f} sec")
    print(f"    Total elapsed:  {sum(dls)+sum(parses):.1f} sec "
          f"({(sum(dls)+sum(parses))/60:.1f} min)")

if not results:
    print("\n  No phenotypes processed successfully. Stopping.")
    raise SystemExit(1)

# ── 4. BUILD AFR Z-SCORE MATRIX ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Building AFR Z-score Matrix")
print("=" * 65)

n_pheno_afr = len(results)
Z_afr = np.zeros((n_pheno_afr, n_snps), dtype=np.float32)

for row_i, res in enumerate(results):
    neff = res['n_eff_AFR']
    for col_j, z in res['z_raw'].items():
        Z_afr[row_i, col_j] = z / np.sqrt(neff)

print(f"  Z_afr matrix shape: {Z_afr.shape}")
print(f"  Value range: [{Z_afr.min():.4f}, {Z_afr.max():.4f}]")
print(f"  Sparsity (fraction zero): {(Z_afr == 0).mean():.4f}")

phenocodes = [r['phenocode'] for r in results]
descriptions = [r['description'] for r in results]

Z_afr_df = pd.DataFrame(Z_afr, columns=target_snp_list)
Z_afr_df.insert(0, 'phenocode', phenocodes)
Z_afr_df.insert(1, 'description', descriptions)
Z_afr_df.to_csv(f'{RESULTS_DIR}/afr_zscore_matrix.csv.gz', index=False,
                compression='gzip')
print(f"  Saved: afr_zscore_matrix.csv.gz ({Z_afr.shape})")

# ── 5. tSVD ON AFR Z-SCORE MATRIX ───────────────────────────────────────────
print("\n" + "=" * 65)
print(f"STEP 5: Truncated SVD on AFR Z-score Matrix (k={K_NMF})")
print("=" * 65)

from sklearn.decomposition import TruncatedSVD
k_eff = min(K_NMF, n_pheno_afr - 1, n_snps)
tsvd = TruncatedSVD(n_components=k_eff, random_state=42)
U_afr = tsvd.fit_transform(Z_afr)
print(f"  U_afr shape: {U_afr.shape}")
print(f"  Explained variance ratio (sum): {tsvd.explained_variance_ratio_.sum():.4f}")
print(f"  Per-component: {np.round(tsvd.explained_variance_ratio_, 4)}")

U_afr_df = pd.DataFrame(U_afr, columns=[f'AFR_SVD_PC{i+1}' for i in range(k_eff)])
U_afr_df.insert(0, 'phenocode', phenocodes)
U_afr_df.insert(1, 'description', descriptions)
U_afr_df.to_csv(f'{RESULTS_DIR}/afr_tsvd_U.csv', index=False)
print(f"  Saved: afr_tsvd_U.csv")

# ── 6. NMF ON AFR Z-SCORE MATRIX ────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"STEP 6: NMF on AFR Z-score Matrix (k={k_eff})")
print("=" * 65)

scaler = MinMaxScaler()
Z_afr_scaled = scaler.fit_transform(Z_afr)

nmf_afr = NMF(n_components=k_eff, init='nndsvda', random_state=42,
               max_iter=500, tol=1e-4)
W_afr = nmf_afr.fit_transform(Z_afr_scaled)
H_afr = nmf_afr.components_
Z_afr_recon = W_afr @ H_afr

ss_total = np.sum((Z_afr_scaled - Z_afr_scaled.mean()) ** 2)
ss_resid = np.sum((Z_afr_scaled - Z_afr_recon) ** 2)
var_explained = 1 - ss_resid / ss_total

print(f"  Converged in {nmf_afr.n_iter_} iterations")
print(f"  Reconstruction error: {nmf_afr.reconstruction_err_:.4f}")
print(f"  Variance explained:   {var_explained:.4f}")
print(f"  W_afr shape: {W_afr.shape}")

W_afr_df = pd.DataFrame(W_afr, columns=[f'AFR_NMF_comp{i+1}' for i in range(k_eff)])
W_afr_df.insert(0, 'phenocode', phenocodes)
W_afr_df.insert(1, 'description', descriptions)
W_afr_df.to_csv(f'{RESULTS_DIR}/afr_nmf_W_loadings.csv', index=False)
print(f"  Saved: afr_nmf_W_loadings.csv")

# ── 7. COMPARE TO EUR FACTORGO ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7: Comparing AFR Factors to EUR FactorGo (per-SNP loading corr)")
print("=" * 65)
print("""
  With only 15 AFR phenotypes, phenotype-level correlation (as in the
  original script) isn't meaningful (n=15 is too small for stable r).
  Instead we compare SNP-loading vectors: correlate each AFR NMF
  component's H (SNP weights) against each EUR FactorGo factor's Zm
  column is not directly comparable (different matrix orientations),
  so instead we report which EUR phenotypes (by phenocode) match our
  15 AFR phenotypes and compare their EUR FactorGo Zm rows to our
  AFR NMF W rows directly (phenotype-loading comparison, n=15).
""")

eur_phenocodes = manifest['phenocode'].values
eur_idx_map = {pc: i for i, pc in enumerate(eur_phenocodes)}

common = [pc for pc in phenocodes if pc in eur_idx_map]
print(f"  AFR subset phenotypes: {len(phenocodes)}")
print(f"  Found in EUR manifest: {len(common)}")

if len(common) >= 3:
    afr_rows = [phenocodes.index(pc) for pc in common]
    eur_rows = [eur_idx_map[pc] for pc in common]

    W_afr_common = W_afr[afr_rows, :]
    Zm_eur_common = Zm_eur.values[eur_rows, :]

    corr = np.zeros((k_eff, 100))
    for i in range(k_eff):
        for j in range(100):
            if W_afr_common[:, i].std() > 0 and Zm_eur_common[:, j].std() > 0:
                r, _ = pearsonr(W_afr_common[:, i], Zm_eur_common[:, j])
                corr[i, j] = r

    best_match = np.argmax(np.abs(corr), axis=1)
    best_corr = np.max(np.abs(corr), axis=1)

    print(f"\n  AFR NMF vs EUR FactorGo (n={len(common)} phenotypes, CAUTION: small n):")
    for i in range(k_eff):
        print(f"    AFR-NMF {i+1:2d}  <-> EUR Factor {best_match[i]+1:3d}  |r|={best_corr[i]:.3f}")

    comp_df = pd.DataFrame({
        'afr_nmf_component': np.arange(1, k_eff+1),
        'best_eur_factorgo_factor': best_match + 1,
        'best_eur_factorgo_corr': best_corr,
    })
    comp_df.to_csv(f'{RESULTS_DIR}/afr_nmf_comparison_table.csv', index=False)
    print(f"\n  Saved: afr_nmf_comparison_table.csv")
    print(f"  NOTE: n={len(common)} is too small for statistically robust")
    print(f"  correlations — report these as exploratory/descriptive only.")
else:
    print("  Too few common phenotypes for comparison.")

# ── 8. FIGURE ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 8: Generating Figure")
print("=" * 65)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#FAFAFA')

# (A) Variance explained: tSVD vs NMF
axes[0].bar(['tSVD\n(cumulative)', 'NMF'],
            [tsvd.explained_variance_ratio_.sum(), var_explained],
            color=['#3498DB', '#E67E22'])
axes[0].set_ylim(0, 1)
axes[0].set_title(f'(A) Variance Explained\n(k={k_eff}, n_pheno={n_pheno_afr})')
axes[0].set_ylabel('Variance explained')

# (B) tSVD PC1 vs PC2
axes[1].scatter(U_afr[:, 0], U_afr[:, 1] if k_eff > 1 else np.zeros(n_pheno_afr),
                s=60, alpha=0.7, color='#9B59B6')
for i, d in enumerate(descriptions):
    axes[1].annotate(d[:15], (U_afr[i, 0], U_afr[i, 1] if k_eff > 1 else 0),
                      fontsize=6)
axes[1].set_xlabel('AFR tSVD PC1')
axes[1].set_ylabel('AFR tSVD PC2')
axes[1].set_title('(B) AFR Phenotypes in\ntSVD Space')

# (C) NMF reconstruction
sample = np.random.choice(Z_afr_scaled.size, min(2000, Z_afr_scaled.size), replace=False)
axes[2].scatter(Z_afr_scaled.flatten()[sample], Z_afr_recon.flatten()[sample],
                alpha=0.2, s=5, color='#2ECC71')
lim = max(Z_afr_scaled.max(), Z_afr_recon.max())
axes[2].plot([0, lim], [0, lim], 'r--', linewidth=1)
axes[2].set_xlabel('Original (scaled)')
axes[2].set_ylabel('NMF Reconstructed')
axes[2].set_title(f'(C) NMF Reconstruction\nVar exp={var_explained:.3f}')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/afr_nmf_main_figure.png', dpi=150,
             bbox_inches='tight', facecolor='#FAFAFA')
plt.close()
print("  Saved: afr_nmf_main_figure.png")

# ── 9. SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 9: Final Summary")
print("=" * 65)
print(f"""
  AFR phenotypes processed:  {n_pheno_afr} (biomarkers, top Neff)
  Z-score matrix shape:      {Z_afr.shape}
  tSVD variance explained:   {tsvd.explained_variance_ratio_.sum():.4f}
  NMF variance explained:    {var_explained:.4f}

  OUTPUT FILES (in '{RESULTS_DIR}/'):
    afr_subset_selected.csv      — which 15 phenotypes were chosen and why
    afr_zscore_matrix.csv.gz      — AFR Z-score matrix {Z_afr.shape}
    afr_tsvd_U.csv                — tSVD phenotype loadings
    afr_nmf_W_loadings.csv         — NMF phenotype loadings
    afr_nmf_comparison_table.csv  — AFR vs EUR FactorGo correlations (n={len(common) if len(common)>=3 else 0})
    afr_nmf_main_figure.png       — summary figure

  CAVEAT FOR YOUR WRITEUP:
  This run uses only {n_pheno_afr} phenotypes (vs EUR's 2,483) due to
  time constraints — full Pan-UKBB per-phenotype files are ~1.5GB each
  and lack pre-indexed AFR-only subsets. Treat results as a feasibility
  pilot, not a definitive AFR-vs-EUR comparison. With more time, the
  full 1,493-phenotype AFR-eligible set (Neff>=100, joined from the
  phenotype manifest) could be processed using the same code structure.
""")
