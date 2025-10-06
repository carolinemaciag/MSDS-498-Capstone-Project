# title: 02_Predictive_Modeling
# author: Jessica Navratil-Strawn
# date: 10/5/2025
# note: This program starts with exploratory data analysis to check for missings, outlier values, and correlation
#       testing. Then after this it predicts whether someone will renew or not
#       Timing of renewal code has not been developed yet.
#------------------------------------------------------------------------------------------------------------------#

# Load Python Libraries
import pandas as pd
import seaborn as sns
import janitor
from toolz import curry
import openpyxl
import os
import snowflake.connector
import numpy as np
from plotnine import ggplot, aes, geom_line, geom_point, labs, theme_minimal
import matplotlib.pyplot as plt
import duckdb
from janitor.functions import clean_names
from scipy import stats
import math
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

# Set up working directory
os.chdir("/Users/jessica.navratil-strawn/Documents/Northwestern/Capstone Project/Data")
print("Current working directory:", os.getcwd())

# Read local lookup
renewal_data = pd.read_csv("renewal_data_deidentified.csv")

#--------------------------------------------------------------------------------------------------------------#
# Look at a summary of the data and check for missing values
#--------------------------------------------------------------------------------------------------------------#

summary = pd.DataFrame({
    "N": renewal_data.count(),
    "Mean": renewal_data.mean(numeric_only=True),
    "Median": renewal_data.median(numeric_only=True),
    "Min": renewal_data.min(numeric_only=True),
    "Max": renewal_data.max(numeric_only=True),
    "Missing": renewal_data.isna().sum()
})

print(summary)

#--------------------------------------------------------------------------------------------------------------#
# Continuous variables convert into bins
#--------------------------------------------------------------------------------------------------------------#

vars_to_bin = [
    'article_views_total', 'guide_views_total', 'benefit_guide_views', 'match_interaction',
    'credit_card_requested', 'expense_created', 'group_session_scheduled', 'logins', 'phone_support', 'provider_finder',
    'partner_agency_referral', 'support_message_thread', 'operational_email_received', 'wheel_login',
    'adoption_article_views', 'other_article_views', 'fertility_article_views', 'gac_article_views',
    'gest_surrogacy_article_views', 'low_t_article_views', 'menopause_article_views', 'parenting_article_views',
    'pregnancy_article_views', 'adoption_guide_views', 'fertility_guide_views', 'gest_surrogacy_guide_views',
    'low_t_guide_views', 'menopause_guide_views', 'parenting_guide_views', 'pregnancy_guide_views']

df = renewal_data  # or df = renewal_data.copy() to avoid in-place mutation

# Set up super-inflated threshold so that if ≥70% are zeros, do not bin
ZERO_INFLATION_THRESH = 0.70  # >=70% zeros among non-missing => super zero-inflated (skip binning)

# Helpers
def _coerce_numeric(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)

def _quartiles(valid: pd.Series):
    q = valid.quantile([0.25, 0.50, 0.75], interpolation="linear")
    return float(q.iloc[0]), float(q.iloc[1]), float(q.iloc[2])

def _fmt_cut(x: float, sig=6) -> str:
    """
    Compact number for column names (up to `sig` significant digits).
    Keeps '.' and '-' (pandas is fine with them). Trims trailing zeros.
    """
    if pd.isna(x):
        return "NA"
    s = f"{x:.{sig}g}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"

# Main
created_columns = {}                 # {var: [list of created dummy columns]}
binned_not_super_zero = []           # collected vars binned into quartiles
not_binned_super_zero_inflated = []  # collected vars skipped due to super zero-inflation
cutpoints_used = {}                  # {var: {"q25":..., "q50":..., "q75":...}}

missing = [v for v in vars_to_bin if v not in df.columns]
if missing:
    print(f"[warn] Skipping missing columns: {missing}")

for v in vars_to_bin:
    if v not in df.columns:
        continue

    s = _coerce_numeric(df[v])
    valid = s.dropna()

    # If no variability or no data, skip
    if valid.empty or valid.nunique() == 1:
        print(f"[info] {v}: insufficient variability; no dummies created.")
        continue

    # Zero-inflation among non-missing
    zero_share = (valid == 0).mean()

    if zero_share >= ZERO_INFLATION_THRESH:
        # Super zero-inflated: DO NOTHING (you already have binary indicators elsewhere)
        not_binned_super_zero_inflated.append(v)
        print(f"[info] {v}: zero-inflated ({zero_share:.1%}). Skipping binning per instructions.")
        continue

    # Not super zero-inflated -> make quartile dummies with ACTUAL NUMERIC CUTS in the column names
    q25, q50, q75 = _quartiles(valid)
    cutpoints_used[v] = {"q25": q25, "q50": q50, "q75": q75}

    # If quartiles collapse (rare without zeros), fall back to rank-based grouping for masks;
    # column names still reflect the *actual* q25/q50/q75 numbers for transparency.
    if not (q25 < q50 < q75):
        ranks = valid.rank(method="average", pct=True)
        m_p0_25   = s.index.isin(ranks.index[ranks <= 0.25])
        m_p26_50  = s.index.isin(ranks.index[(ranks > 0.25) & (ranks <= 0.50)])
        m_p51_75  = s.index.isin(ranks.index[(ranks > 0.50) & (ranks <= 0.75)])
        m_p75_100 = s.index.isin(ranks.index[ranks > 0.75])
    else:
        m_p0_25   = s <= q25
        m_p26_50  = (s > q25) & (s <= q50)
        m_p51_75  = (s > q50) & (s <= q75)
        m_p75_100 = s > q75

    q25n = _fmt_cut(q25)
    q50n = _fmt_cut(q50)
    q75n = _fmt_cut(q75)

    # Column names embed the *numeric* ranges
    c1 = f"{v}_le_{q25n}"
    c2 = f"{v}_gt_{q25n}_le_{q50n}"
    c3 = f"{v}_gt_{q50n}_le_{q75n}"
    c4 = f"{v}_gt_{q75n}"

    # Create 0/1 dummies (Int8); NaNs -> 0
    df[c1] = pd.Series(m_p0_25,   index=df.index).fillna(False).astype("Int8")
    df[c2] = pd.Series(m_p26_50,  index=df.index).fillna(False).astype("Int8")
    df[c3] = pd.Series(m_p51_75,  index=df.index).fillna(False).astype("Int8")
    df[c4] = pd.Series(m_p75_100, index=df.index).fillna(False).astype("Int8")

    created_columns[v] = [c1, c2, c3, c4]
    binned_not_super_zero.append(v)

# Summary
print("\n===== Binning Summary =====")
print(f"Binned (not super zero-inflated) [{len(binned_not_super_zero)}]:")
if binned_not_super_zero:
    for v in binned_not_super_zero:
        qs = cutpoints_used.get(v, {})
        print(f"  - {v}: q25={_fmt_cut(qs.get('q25', np.nan))}, q50={_fmt_cut(qs.get('q50', np.nan))}, q75={_fmt_cut(qs.get('q75', np.nan))}")
else:
    print("  (none)")

print(f"\nSkipped (super zero-inflated) [{len(not_binned_super_zero_inflated)}]:")
if not_binned_super_zero_inflated:
    print("  " + ", ".join(not_binned_super_zero_inflated))
else:
    print("  (none)")

# These variables were binned:
# benefit_guide_views, expense_created, logins, provider_finder,
# support_message_thread, operational_email_received

#--------------------------------------------------------------------------------------------------------------#
# Correlation test
#--------------------------------------------------------------------------------------------------------------#

print(renewal_data.columns.tolist())

TARGET_BIN  = "renewal"
TARGET_CONT = "trunc_months_between_unlocks"
ID_VARS     = ["employee_id_deid","company_id_deid"]

CONT_VARS = ['trunc_current_annual_benefit_maximum','trunc_current_lifetime_benefit_maximum', 'trunc_employee_age',]

BIN_VARS = ['is_case_rate', 'allows_registration_emails', 'up_for_first_renewal', 'deductible_status',
    'co_pay_or_co_insurance', 'is_medically_necessary_preservation_coverage_active', 'is_adoption_coverage_active',
    'is_elective_preservation_coverage_active', 'is_art_with_infertility_diagnosis_coverage_active',
    'is_elective_art_coverage_active', 'is_gender_affirming_care_coverage_active', 'is_menopause_coverage_active',
    'is_low_t_coverage_active', 'is_pregnancy_and_postpartum_coverage_active', 'is_gestational_carrier_coverage_active',
    'is_doula_expense_coverage_active', 'is_childbirth_class_coverage_active', 'is_milk_shipping_coverage_active',
    'sex', 'country_US', 'market_segment_MID', 'market_segment_SMB', 'market_segment_STRAT',
    #'market_segment_NATIONAL',
    'first_unlock_journey_ADOPTION', 'first_unlock_journey_ASSISTED_REPRODUCTION', 'first_unlock_journey_EXPLORING',
    'first_unlock_journey_GENDER_AFFIRMING_CARE', 'first_unlock_journey_GESTATIONAL',
    'first_unlock_journey_LOW_TESTOSTERONE', 'first_unlock_journey_MENOPAUSE', 'first_unlock_journey_PARENTING',
    'first_unlock_journey_PREGNANT', 'first_unlock_journey_PRESERVATION', 'first_unlock_journey_SOMETHING_ELSE',
    'first_unlock_journey_TRY_PREGNANT', 'first_unlock_journey_unknown', 'program_type_CORE', 'program_type_PRO',
    'outl_current_annual_benefit_maximum', 'outl_current_lifetime_benefit_maximum', 'outl_employee_age',
    'match_interaction_flag', 'credit_card_requested_flag', 'group_session_scheduled_flag',

    # Commented these flags out for now, I would test model accuracy with them in and out of model
    'logins_flag', 'expense_created_flag', 'benefit_guide_views_flag', 'provider_finder_flag', 'support_message_thread_flag', 'operational_email_received_flag',

    'phone_support_flag', 'partner_agency_referral_flag', 'wheel_login_flag',
    'adoption_article_views_flag', 'other_article_views_flag', 'fertility_article_views_flag', 'gac_article_views_flag',
    'gest_surrogacy_article_views_flag', 'low_t_article_views_flag', 'menopause_article_views_flag',
    'parenting_article_views_flag', 'pregnancy_article_views_flag', 'adoption_guide_views_flag',
    'fertility_guide_views_flag', 'gest_surrogacy_guide_views_flag', 'low_t_guide_views_flag',
    'menopause_guide_views_flag', 'parenting_guide_views_flag', 'pregnancy_guide_views_flag']

# Commented out these flags because the binary flags outperform the binned variables
BINNED_VARS = [
# 'benefit_guide_views_gt_0_le_4', 'benefit_guide_views_gt_4_le_16', 'benefit_guide_views_gt_16',
# 'expense_created_gt_0_le_0', 'expense_created_gt_0_le_2', 'expense_created_gt_2',
# 'logins_gt_1_le_4', 'logins_gt_4_le_14', 'logins_gt_14',
# 'provider_finder_gt_0_le_0', 'provider_finder_gt_0_le_3', 'provider_finder_gt_3',
# 'support_message_thread_gt_0_le_0', 'support_message_thread_gt_0_le_2', 'support_message_thread_gt_2',
# 'operational_email_received_gt_1_le_2', 'operational_email_received_gt_2_le_4', 'operational_email_received_gt_4'
# Taking out the zero category because of linear combination
# 'benefit_guide_views_le_0','expense_created_le_0','logins_le_1','provider_finder_le_0', 'operational_email_received_le_1', 'support_message_thread_le_0',
]

# Merge CONT_VARS + cont_vars_to_bin (dedupe, preserve order)
def _unique_preserve(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

ALL_CONT_VARS = _unique_preserve(CONT_VARS + BINNED_VARS)

# Your dataframe
df = renewal_data.copy()

# ----- helpers -----
def _coerce_numeric(series):
    """Coerce a pandas Series to numeric, keeping NaN for non-convertible."""
    return pd.to_numeric(series, errors='coerce')

def _valid_pair(num_a, num_b):
    """Ensure both vectors have at least 2 non-NaN points and nonzero variance."""
    if len(num_a) < 2 or len(num_b) < 2:
        return False
    # Need some variability for Pearson to work
    return (np.nanstd(num_a) > 0) and (np.nanstd(num_b) > 0)

# ----- Correlation block -----
def corr_block(target, predictors, predictor_type, method_label):
    out = []
    for v in predictors:
        if v not in df.columns or target not in df.columns:
            continue

        pair = df[[target, v]].copy()
        # Coerce both columns to numeric for Pearson calc
        pair.iloc[:, 0] = _coerce_numeric(pair.iloc[:, 0])
        pair.iloc[:, 1] = _coerce_numeric(pair.iloc[:, 1])
        pair = pair.dropna()

        if pair.shape[0] < 2:
            continue

        x = pair.iloc[:, 0].values
        y = pair.iloc[:, 1].values

        # Guard against zero variance
        if not _valid_pair(x, y):
            continue

        r, p = stats.pearsonr(x, y)
        out.append({
            "target": target,
            "predictor": v,
            "predictor_type": predictor_type,
            "method": method_label,
            "N": pair.shape[0],
            "coef": r,
            "p_value": p
        })
    return out

# ----- Build the correlation table -----
rows = []
# TARGET_BIN vs continuous (point-biserial via Pearson)
rows += corr_block(TARGET_BIN, ALL_CONT_VARS, "continuous", "point-biserial (Pearson)")
# TARGET_BIN vs binary (phi via Pearson on 0/1)
rows += corr_block(TARGET_BIN, BIN_VARS, "binary", "phi (Pearson on 0/1)")
# TARGET_CONT vs continuous (Pearson)
rows += corr_block(TARGET_CONT, ALL_CONT_VARS, "continuous", "Pearson")
# TARGET_CONT vs binary (point-biserial via Pearson)
rows += corr_block(TARGET_CONT, BIN_VARS, "binary", "point-biserial (Pearson)")

corr_table = pd.DataFrame(rows).sort_values(["target","p_value","predictor"])

# Visualization (color-coded heatmaps)
# Strength labels (by absolute r)
def strength_label(r):
    ar = abs(r)
    if ar < 0.30:
        return "Weak"
    elif ar < 0.50:
        return "Moderate"
    else:
        return "Strong"

# Significance stars
def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

corr_table["abs_coef"]  = corr_table["coef"].abs()
corr_table["strength"]  = corr_table["coef"].apply(strength_label)
corr_table["sig"]       = corr_table["p_value"].apply(sig_stars)
corr_table["label"]     = corr_table["coef"].round(3).astype(str) + corr_table["sig"]

# Sort within each target by |r|
corr_table = corr_table.sort_values(["target", "abs_coef", "predictor"], ascending=[True, False, True])

# ---------- Build pivoted matrices (values = r, text = r + stars) ----------
def make_pivots(df_in, target):
    sub = df_in[df_in["target"] == target].copy()
    sub = sub.sort_values("abs_coef", ascending=False)  # order rows by |r|
    mat = sub.pivot(index="predictor", columns="target", values="coef")
    txt = sub.pivot(index="predictor", columns="target", values="label")
    strength = sub.pivot(index="predictor", columns="target", values="strength")
    return mat, txt, strength

mat_bin,  txt_bin,  str_bin  = make_pivots(corr_table, TARGET_BIN)
mat_cont, txt_cont, str_cont = make_pivots(corr_table, TARGET_CONT)

# ---------- Plotting utility (matplotlib only) ----------
def plot_heatmap(values_df, text_df, title):
    # Ensure annotation text has no NaNs
    text_df = text_df.fillna("")

    data = values_df.values.astype(float)
    n_rows = data.shape[0]
    fig_h = max(6, min(0.4 * n_rows + 2, 30))  # auto-size
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=120)

    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)

    # Axes/labels
    ax.set_title(title, pad=12)
    ax.set_yticks(range(values_df.shape[0]))
    ax.set_yticklabels(values_df.index, fontsize=9)
    ax.set_xticks(range(values_df.shape[1]))
    ax.set_xticklabels(values_df.columns, fontsize=10)

    # Grid
    ax.set_xticks(np.arange(-.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate each cell: r + stars + strength (compute strength inline to avoid dtype issues)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            r_txt = text_df.iloc[i, j]
            stg   = strength_label(data[i, j])  # compute from float value
            ax.text(j, i, f"{r_txt}\n{stg}", ha="center", va="center", fontsize=8, color="black")

    # Colorbar and legend
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r (sign & magnitude)")

    ax.text(0, -0.07, "Strength thresholds (by |r|): Weak < 0.30, Moderate 0.30–0.49, Strong ≥ 0.50",
            transform=ax.transAxes, fontsize=9)

    plt.tight_layout()
    plt.show()

# ---------- Draw the two heatmaps ----------
plot_heatmap(mat_bin,  txt_bin,  "Correlation with TARGET_BIN = 'renewal'")
plot_heatmap(mat_cont, txt_cont, "Correlation with TARGET_CONT = 'trunc_months_between_unlocks'")

# Excel Export with conditional formatting
def export_corr_matrix_to_excel(df_in, path="correlation_matrix.xlsx"):
    import pandas as pd
    from xlsxwriter.utility import xl_col_to_name

    out = df_in.copy()

    # Ensure coef is numeric and abs_coef exists
    out["coef"] = pd.to_numeric(out["coef"], errors="coerce")
    if "abs_coef" not in out.columns:
        out["abs_coef"] = out["coef"].abs()

    # Keep a tidy column order; include abs_coef for sorting then hide it
    cols = ["target","predictor","predictor_type","coef","p_value","sig","strength","N","abs_coef"]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].sort_values(["target","abs_coef","predictor"], ascending=[True, False, True])

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        out.to_excel(writer, index=False, sheet_name="corr")
        wb = writer.book
        ws = writer.sheets["corr"]

        # Locate dynamic column indexes
        coef_idx = out.columns.get_loc("coef")
        pval_idx = out.columns.get_loc("p_value") if "p_value" in out.columns else None
        abs_idx  = out.columns.get_loc("abs_coef") if "abs_coef" in out.columns else None

        # Excel row/col math
        n_rows = len(out)
        start_row = 1  # data starts on row 2 (0-based -> +1)
        end_row = start_row + n_rows - 1

        # Build Excel range for coef column (e.g., "D2:D100")
        coef_col_letter = xl_col_to_name(coef_idx)
        coef_range = f"{coef_col_letter}{start_row+1}:{coef_col_letter}{end_row+1}"

        # Conditional formats by |r|
        ws.conditional_format(coef_range, {
            "type": "cell", "criteria": "between", "minimum": -0.30, "maximum": 0.30,
            "format": wb.add_format({"bg_color": "#F0F0F0"})
        })
        ws.conditional_format(coef_range, {
            "type": "cell", "criteria": ">=", "value": 0.30,
            "format": wb.add_format({"bg_color": "#FFD9B3"})
        })
        ws.conditional_format(coef_range, {
            "type": "cell", "criteria": "<=", "value": -0.30,
            "format": wb.add_format({"bg_color": "#B3D7FF"})
        })
        ws.conditional_format(coef_range, {
            "type": "cell", "criteria": ">=", "value": 0.50,
            "format": wb.add_format({"bg_color": "#FFB380"})
        })
        ws.conditional_format(coef_range, {
            "type": "cell", "criteria": "<=", "value": -0.50,
            "format": wb.add_format({"bg_color": "#80B2FF"})
        })

        # Number formats and widths
        coef_numfmt = wb.add_format({"num_format": "0.000"})
        ws.set_column(coef_idx, coef_idx, 10, coef_numfmt)
        if pval_idx is not None:
            p_numfmt = wb.add_format({"num_format": "0.0000"})
            ws.set_column(pval_idx, pval_idx, 12, p_numfmt)

        # Helpful widths for text columns if present
        if "target" in out.columns:    ws.set_column(out.columns.get_loc("target"),    out.columns.get_loc("target"),    16)
        if "predictor" in out.columns: ws.set_column(out.columns.get_loc("predictor"), out.columns.get_loc("predictor"), 34)
        if "predictor_type" in out.columns:
            ws.set_column(out.columns.get_loc("predictor_type"), out.columns.get_loc("predictor_type"), 16)

        # Hide abs_coef helper column
        if abs_idx is not None:
            ws.set_column(abs_idx, abs_idx, None, None, {"hidden": True})

        # Freeze header row & add autofilter
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, n_rows, len(out.columns)-1)

# Run export (writes correlation_matrix.xlsx to CWD)
export_corr_matrix_to_excel(corr_table)

#--------------------------------------------------------------------------------------------------------------#
# Boxplots - Continuous variables for renewal (yes/no)
#--------------------------------------------------------------------------------------------------------------#

# ---- Layout ----
n = len(CONT_VARS)
ncols = 3 if n >= 3 else n
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows), sharey=False)
if n == 1:
    axes = np.array([axes])
axes = axes.ravel()

median_proxy = None
mean_proxy = None

# ---- Plot each variable ----
for i, var in enumerate(CONT_VARS):
    ax = axes[i]
    grp0 = df.loc[df[TARGET_BIN] == 0, var].dropna()
    grp1 = df.loc[df[TARGET_BIN] == 1, var].dropna()

    bp = ax.boxplot(
        [grp0.values, grp1.values],
        tick_labels=["No", "Yes"],   # <- updated for Matplotlib 3.9+
        showmeans=True,
        meanline=True
    )

    ax.set_title(var)
    ax.set_xlabel(f"{TARGET_BIN} (No/Yes)")
    ax.set_ylabel(var)

    # Capture legend proxies once using actual drawn styles
    if median_proxy is None and bp.get('medians'):
        med = bp['medians'][0]
        median_proxy = Line2D([0],[0], linestyle=med.get_linestyle(),
                              linewidth=med.get_linewidth(), color=med.get_color(),
                              label="Median")
    if mean_proxy is None and bp.get('means'):
        meanl = bp['means'][0]
        mean_proxy = Line2D([0],[0], linestyle=meanl.get_linestyle(),
                            linewidth=meanl.get_linewidth(), color=meanl.get_color(),
                            label="Mean")

# Hide any unused axes (if grid bigger than variables)
for j in range(len(CONT_VARS), len(axes)):
    fig.delaxes(axes[j])

# ---- Legend at bottom, then master title, then layout/show (all once) ----
handles = [h for h in [median_proxy, mean_proxy] if h is not None]
if handles:
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.02))

fig.suptitle("Box plots of continuous variables by renewal", y=0.98, fontsize=14)

# Leave room at bottom for legend and top for title
plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.92])  # [left, bottom, right, top]
plt.show()

# SUMMARY: The boxplots suggest that higher annual and lifetime maximums are
# associated with a higher likelihood of renewal

#--------------------------------------------------------------------------------------------------------------#
# Barplots - Binary variables for renewal (yes/no)
#--------------------------------------------------------------------------------------------------------------#

# --- Helper to plot a chunk of up to 12 variables (UNCHANGED) ---
def plot_binvars_page(vars_chunk, page_idx=1):
    n = len(vars_chunk)
    ncols, nrows = 3, 4  # 3x4 = 12
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 3.8*nrows), sharey=True)
    axes = axes.ravel()

    for i, var in enumerate(vars_chunk):
        ax = axes[i]
        sub = df[[var, TARGET_BIN]].dropna()
        if sub.empty:
            ax.set_title(f"{var} (no data)")
            ax.axis("off")
            continue

        # Renewal rate by var value (assumes 0/1 coding)
        grp = sub.groupby(var)[TARGET_BIN].agg(['mean','count']).reindex([0,1])  # order 0,1 if present
        means = grp['mean']
        counts = grp['count']  # kept for consistency, but not plotted

        # Build bars
        bars = ax.bar(range(len(means)), means.fillna(0.0).values)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels([str(idx) for idx in means.index], rotation=0)
        ax.set_title(var)
        ax.set_xlabel(f"{var} (0/1)")
        if i % ncols == 0:
            ax.set_ylabel("Renewal rate")

        # (Removed N annotations to reduce chart clutter)

    # Remove any leftover empty axes
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Renewal rate by binary variables (page {page_idx})", y=0.98, fontsize=14)
    plt.tight_layout(rect=[0.04, 0.04, 0.98, 0.94])  # leave space for title
    plt.show()

# --- NEW: combine BIN_VARS + binned_vars and use that everywhere below ---
ALL_BIN_VARS = list(dict.fromkeys(BIN_VARS + BINNED_VARS))

# --- Split ALL_BIN_VARS into pages of 12 and plot each page ---
present = [v for v in ALL_BIN_VARS if v in df.columns]
missing = sorted(set(ALL_BIN_VARS) - set(present))
if missing:
    print(f"Skipping missing columns ({len(missing)}): {missing}")

page_size = 12
num_pages = math.ceil(len(present) / page_size)

for p in range(num_pages):
    chunk = present[p*page_size:(p+1)*page_size]
    plot_binvars_page(chunk, page_idx=p+1)

#--------------------------------------------------------------------------------------------------------------#
# Logistic Regression - Renewal (yes/no)
#--------------------------------------------------------------------------------------------------------------#

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import statsmodels.api as sm

# Keep only features available in df
FULL_FEATURES = [c for c in (CONT_VARS + BIN_VARS + BINNED_VARS) if c in df.columns]

# Train / Test Split
cols_needed = [TARGET_BIN] + list(set(FULL_FEATURES))
dfm = df[cols_needed].dropna()
X_full = dfm[FULL_FEATURES]
y = dfm[TARGET_BIN].astype(int)

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.30, random_state=42, stratify=y)

# -----------------------------------------------------------------------------------------------
# Full Logistic Regression
# -----------------------------------------------------------------------------------------------

full_clf = LogisticRegression(solver='liblinear', max_iter=2000)
full_clf.fit(X_train_full, y_train)
full_pred_proba = full_clf.predict_proba(X_test_full)[:,1]
full_auc = roc_auc_score(y_test, full_pred_proba)
full_acc = accuracy_score(y_test, (full_pred_proba >= 0.5).astype(int))

coef_table = pd.DataFrame({
    "variable": X_train_full.columns,
    "coefficient": full_clf.coef_[0],
    "odds_ratio": np.exp(full_clf.coef_[0])})
print(coef_table)

# -----------------------------------------------------------------------------------------------
# Stepwise Logistic Regression (p<0.10 using statsmodels Logit)
# -----------------------------------------------------------------------------------------------

def fit_sm_logit(X, y):
    Xc = sm.add_constant(X, has_constant='add')
    model = sm.Logit(y, Xc, missing='drop')
    return model.fit(disp=0)

def stepwise_logit(X, y, p_enter=0.10, p_remove=0.20, max_iter=100, start_with_full=False):
    remaining = list(X.columns)
    selected = list(X.columns) if start_with_full else []
    changed = True
    it = 0

    while changed and it < max_iter:
        changed = False
        it += 1

        # ----- Forward step (adds best candidate with p < p_enter) -----
        candidates = list(set(remaining) - set(selected))
        if candidates:
            pvals = pd.Series(index=candidates, dtype=float)
            for c in candidates:
                cols = selected + [c]
                try:
                    res = fit_sm_logit(X[cols], y)
                    pvals[c] = res.pvalues.get(c, np.nan)
                except Exception:
                    pvals[c] = np.nan
            if not pvals.empty and pvals.min() < p_enter:
                best = pvals.idxmin()
                selected.append(best)
                changed = True

        # ----- Backward step (drops any selected with p >= p_remove) -----
        if selected:
            try:
                res = fit_sm_logit(X[selected], y)
                pv = res.pvalues.drop('const', errors='ignore')
                worst_p = pv.max()
                if worst_p >= p_remove:
                    worst = pv.idxmax()
                    selected.remove(worst)
                    changed = True
            except Exception:
                # If the fit fails (e.g., separation/collinearity), try removing
                # the last-added variable to recover.
                if selected:
                    selected.pop()
                    changed = True

    return selected

stepwise_selected = stepwise_logit(X_train_full, y_train, p_enter=0.10, p_remove=0.20)
# Fit final sklearn model on selected vars (for consistent ROC/ACC)
X_train_step = X_train_full[stepwise_selected]
X_test_step  = X_test_full[stepwise_selected]
step_clf = LogisticRegression(solver='liblinear', max_iter=2000)
step_clf.fit(X_train_step, y_train)
step_pred_proba = step_clf.predict_proba(X_test_step)[:,1]
step_auc = roc_auc_score(y_test, step_pred_proba)
step_acc = accuracy_score(y_test, (step_pred_proba >= 0.5).astype(int))

# -----------------------------------------------------------------------------------------------
# SUMMARY Logistic Regression
# -----------------------------------------------------------------------------------------------

print("\n=== Model Comparison (Test Set) ===")
print(f"Full model:        AUC={full_auc:.3f}  ACC@0.5={full_acc:.3f}  (#features={X_train_full.shape[1]})")
print(f"Stepwise (p<0.10): AUC={step_auc:.3f}  ACC@0.5={step_acc:.3f}  (#features={len(stepwise_selected)})")

print("\nStepwise selected features:")
print(stepwise_selected)

# -----------------------------------------------------------------------------------------------
# ROC Curve (Test Set)
# -----------------------------------------------------------------------------------------------

fpr_full, tpr_full, _ = roc_curve(y_test, full_pred_proba)
fpr_step, tpr_step, _ = roc_curve(y_test, step_pred_proba)

plt.figure(figsize=(7,6))
plt.plot(fpr_full, tpr_full, label=f"Full (AUC={full_auc:.3f})")
plt.plot(fpr_step, tpr_step, label=f"Stepwise p<0.10 (AUC={step_auc:.3f})")
plt.plot([0,1], [0,1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Logistic Models (Test Set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------#
# Decision Tree - Renewal (yes/no)
#--------------------------------------------------------------------------------------------------------------#

# ===== Fast Decision Tree (tuning + eval + ROC + importances) =====
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.tree import DecisionTreeClassifier

# ----------------- Safety checks -----------------
def _check_data(X, y, name="X"):
    X = np.asarray(X)
    y = np.asarray(y)
    if np.isnan(X).any():
        raise ValueError(f"{name} contains NaNs; impute or drop before fitting.")
    if np.unique(y).size < 2:
        raise ValueError("y has only one class; need both classes present.")

# ----------------- Fast tuner: Decision Tree -----------------
def tune_tree_fast(X_train, y_train, random_state=42, n_iter=40, cv_splits=3, verbose=2):
    _check_data(X_train, y_train, "X_train (tree)")
    # Data-driven pruning strengths
    try:
        path = DecisionTreeClassifier(random_state=random_state).cost_complexity_pruning_path(X_train, y_train)
        alphas = np.unique(np.round(path.ccp_alphas, 6))
        if alphas.size > 0:
            q = np.unique(np.quantile(alphas, [0.0, 0.25, 0.5, 0.75, 1.0]))
            ccp_choices = np.unique(np.clip(q, 0.0, None)).tolist()
        else:
            ccp_choices = [0.0, 1e-4, 5e-4, 1e-3]
    except Exception:
        ccp_choices = [0.0, 1e-4, 5e-4, 1e-3]

    param_dist = {
        "max_depth": [3, 5, 7, 9, None],
        "min_samples_split": [2, 10, 20, 50],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "max_features": ["sqrt", "log2", None],
        "ccp_alpha": ccp_choices,
    }

    base = DecisionTreeClassifier(
        random_state=random_state,
        # class_weight="balanced",  # uncomment if classes are imbalanced
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=verbose,
        random_state=random_state,
        error_score="raise",
    )
    t0 = time.time()
    rs.fit(X_train, y_train)
    print(f"[tune_tree_fast] {rs.n_iter} configs × {cv_splits}-fold in {time.time()-t0:.1f}s; best CV AUC={rs.best_score_:.4f}")
    return rs.best_estimator_, rs.best_params_, rs.best_score_

# ======================= FIT & EVALUATE =======================
# Expect these to already exist in your session:
# X_train_full, X_test_full, y_train, y_test
# Optionally: X_train_step/X_test_step, X_train_imp/X_test_imp

results = []    # (name, AUC, ACC, cols, model)
all_rocs = []   # (label, fpr, tpr)

# ----- Full features -----
tree_full, bp_full, cvauc_full = tune_tree_fast(X_train_full, y_train)
proba_full_tree = tree_full.predict_proba(X_test_full)[:, 1]
auc_full_tree = roc_auc_score(y_test, proba_full_tree)
acc_full_tree = accuracy_score(y_test, (proba_full_tree >= 0.5).astype(int))
results.append(("Tree - Full", auc_full_tree, acc_full_tree, X_train_full.columns, tree_full))
fpr_tree_full, tpr_tree_full, _ = roc_curve(y_test, proba_full_tree)
all_rocs.append((f"Tree Full (AUC={auc_full_tree:.3f})", fpr_tree_full, tpr_tree_full))

# ----- Stepwise-selected (if present) -----
if 'X_train_step' in locals() and X_train_step.shape[1] > 0:
    tree_step, bp_step, cvauc_step = tune_tree_fast(X_train_step, y_train)
    proba_step_tree = tree_step.predict_proba(X_test_step)[:, 1]
    auc_step_tree = roc_auc_score(y_test, proba_step_tree)
    acc_step_tree = accuracy_score(y_test, (proba_step_tree >= 0.5).astype(int))
    results.append(("Tree - Stepwise", auc_step_tree, acc_step_tree, X_train_step.columns, tree_step))
    fpr_tree_step, tpr_tree_step, _ = roc_curve(y_test, proba_step_tree)
    all_rocs.append((f"Tree Stepwise (AUC={auc_step_tree:.3f})", fpr_tree_step, tpr_tree_step))
else:
    print("Stepwise set is empty — skipping tuned tree for Stepwise.")

# ----------------- Print comparisons -----------------
print("\n=== Tuned Decision Trees (Test Set) ===")
for name, aucv, accv, cols, model in results:
    print(f"{name:18s} AUC={aucv:.3f}  ACC@0.5={accv:.3f}  (#features={len(cols)})")

# ----------------- Feature importances -----------------
def print_importances(name, model, cols, top_n=20):
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"variable": list(cols), "importance": model.feature_importances_}) \
                .sort_values("importance", ascending=False).head(top_n)
        print(f"\n{name} — Top {top_n} Feature Importances")
        print(imp.to_string(index=False))

print_importances("Tree - Full", results[0][4], results[0][3])
if len(results) > 1 and results[1][0] == "Tree - Stepwise":
    print_importances("Tree - Stepwise", results[1][4], results[1][3])

# ----------------- ROC plot (trees only) -----------------
plt.figure(figsize=(7,6))
for label, fpr, tpr in all_rocs:
    plt.plot(fpr, tpr, label=label)
plt.plot([0,1],[0,1],'--',label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC — Tuned Decision Trees (Test Set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------#
# Gradient Boosting - Renewal (yes/no)
#--------------------------------------------------------------------------------------------------------------#

# ===== Fast Gradient Boosting (tuning + eval + ROC + importances) =====
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.ensemble import GradientBoostingClassifier

# ----------------- Safety checks -----------------
def _check_data(X, y, name="X"):
    X = np.asarray(X)
    y = np.asarray(y)
    if np.isnan(X).any():
        raise ValueError(f"{name} contains NaNs; impute or drop before fitting.")
    if np.unique(y).size < 2:
        raise ValueError("y has only one class; need both classes present.")

# ----------------- Helper: log-uniform sampler (fallback if SciPy absent) -----------------
try:
    from scipy.stats import loguniform as _scipy_loguniform


    def _sample_learning_rates(n, a=0.01, b=0.2, seed=42):
        rng = np.random.default_rng(seed)
        return _scipy_loguniform(a, b).rvs(size=n, random_state=rng)
except Exception:
    def _sample_learning_rates(n, a=0.01, b=0.2, seed=42):
        rng = np.random.default_rng(seed)
        return np.exp(rng.uniform(np.log(a), np.log(b), size=n))

# ----------------- Fast tuner: Gradient Boosting -----------------
def tune_gb_fast(X_train, y_train, random_state=42, n_iter=40, cv_splits=3, verbose=2):
    _check_data(X_train, y_train, "X_train (GB)")

    # Pre-sample learning rates so we don't depend on scipy's distributions
    lr_candidates = _sample_learning_rates(max(n_iter, 40), a=0.01, b=0.2, seed=random_state)
    lr_candidates = sorted(set(np.round(lr_candidates, 5)))[:max(n_iter, 40)]  # unique & clipped

    param_dist = {
        "n_estimators": [120, 180, 240, 300],
        "learning_rate": lr_candidates,
        "max_depth": [2, 3, 4],
        "min_samples_leaf": [1, 5, 10],
        "subsample": [0.7, 0.85, 1.0],
        "max_features": ["sqrt", "log2", None],
    }

    base = GradientBoostingClassifier(
        random_state=random_state,
        n_iter_no_change=10,  # early stopping
        validation_fraction=0.1,
        tol=1e-4
    )

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=verbose,
        random_state=random_state,
        error_score="raise",
    )

    t0 = time.time()
    rs.fit(X_train, y_train)
    print(
        f"[tune_gb_fast] {rs.n_iter} configs × {cv_splits}-fold in {time.time() - t0:.1f}s; best CV AUC={rs.best_score_:.4f}")
    return rs.best_estimator_, rs.best_params_, rs.best_score_

# ======================= FIT & EVALUATE =======================
# Expect these to already exist in your session:
# X_train_full, X_test_full, y_train, y_test
# Optionally: X_train_step/X_test_step, X_train_imp/X_test_imp

gb_results = []  # (name, AUC, ACC, cols, model)
all_rocs = []  # (label, fpr, tpr)

# ----- Full features -----
gb_full, gb_bp_full, gb_cvauc_full = tune_gb_fast(X_train_full, y_train)
proba_full_gb = gb_full.predict_proba(X_test_full)[:, 1]
auc_full_gb = roc_auc_score(y_test, proba_full_gb)
acc_full_gb = accuracy_score(y_test, (proba_full_gb >= 0.5).astype(int))
gb_results.append(("GB - Full", auc_full_gb, acc_full_gb, X_train_full.columns, gb_full))
fpr_gb_full, tpr_gb_full, _ = roc_curve(y_test, proba_full_gb)
all_rocs.append((f"GB Full (AUC={auc_full_gb:.3f})", fpr_gb_full, tpr_gb_full))

# ----- Stepwise-selected (if present) -----
if 'X_train_step' in locals() and X_train_step.shape[1] > 0:
    gb_step, gb_bp_step, gb_cvauc_step = tune_gb_fast(X_train_step, y_train)
    proba_step_gb = gb_step.predict_proba(X_test_step)[:, 1]
    auc_step_gb = roc_auc_score(y_test, proba_step_gb)
    acc_step_gb = accuracy_score(y_test, (proba_step_gb >= 0.5).astype(int))
    gb_results.append(("GB - Stepwise", auc_step_gb, acc_step_gb, X_train_step.columns, gb_step))
    fpr_gb_step, tpr_gb_step, _ = roc_curve(y_test, proba_step_gb)
    all_rocs.append((f"GB Stepwise (AUC={auc_step_gb:.3f})", fpr_gb_step, tpr_gb_step))
else:
    print("Stepwise set is empty — skipping tuned GB for Stepwise.")

# ----------------- Print comparisons -----------------
print("\n=== Tuned Gradient Boosting (Test Set) ===")
for name, aucv, accv, cols, model in gb_results:
    print(f"{name:18s} AUC={aucv:.3f}  ACC@0.5={accv:.3f}  (#features={len(cols)})")

# ----------------- Feature importances -----------------
def print_importances(name, model, cols, top_n=20):
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"variable": list(cols), "importance": model.feature_importances_}) \
            .sort_values("importance", ascending=False).head(top_n)
        print(f"\n{name} — Top {top_n} Feature Importances")
        print(imp.to_string(index=False))

print_importances("GB - Full", gb_results[0][4], gb_results[0][3])
if len(gb_results) > 1 and gb_results[1][0] == "GB - Stepwise":
    print_importances("GB - Stepwise", gb_results[1][4], gb_results[1][3])

# ----------------- ROC plot (GB only) -----------------
plt.figure(figsize=(7, 6))
for label, fpr, tpr in all_rocs:
    plt.plot(fpr, tpr, label=label)
plt.plot([0, 1], [0, 1], '--', label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC — Tuned Gradient Boosting (Test Set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------#
# Random Forest - Renewal (yes/no)
#--------------------------------------------------------------------------------------------------------------#

# ===== Fast Random Forest (tuning + eval + ROC + importances) =====
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

# ----------------- Safety checks -----------------
def _check_data(X, y, name="X"):
    X = np.asarray(X)
    y = np.asarray(y)
    if np.isnan(X).any():
        raise ValueError(f"{name} contains NaNs; impute or drop before fitting.")
    if np.unique(y).size < 2:
        raise ValueError("y has only one class; need both classes present.")

# ----------------- Fast tuner: Random Forest -----------------
def tune_rf_fast(X_train, y_train, random_state=42, n_iter=40, cv_splits=3, verbose=2):
    _check_data(X_train, y_train, "X_train (RF)")
    # Focused, high-impact hyperparams
    param_dist = {
        "n_estimators": [300, 500, 800, 1200],
        "max_depth": [None, 8, 12, 16, 24],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5],  # 0.5 uses half of features
        "bootstrap": [True, False],
    }

    base = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,               # parallelize tree building
        oob_score=False,         # set True only if bootstrap=True & not in CV
        # class_weight="balanced" # uncomment if classes are imbalanced
    )

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=verbose,
        random_state=random_state,
        error_score="raise",
    )

    t0 = time.time()
    rs.fit(X_train, y_train)
    print(f"[tune_rf_fast] {rs.n_iter} configs × {cv_splits}-fold in {time.time()-t0:.1f}s; best CV AUC={rs.best_score_:.4f}")
    return rs.best_estimator_, rs.best_params_, rs.best_score_

# ======================= FIT & EVALUATE =======================
# Expect these to already exist in your session:
# X_train_full, X_test_full, y_train, y_test
# Optionally: X_train_step/X_test_step, X_train_imp/X_test_imp

rf_results = []  # (name, AUC, ACC, cols, model)
all_rocs = []    # (label, fpr, tpr)

# ----- Full features -----
rf_full, rf_bp_full, rf_cvauc_full = tune_rf_fast(X_train_full, y_train)
proba_full_rf = rf_full.predict_proba(X_test_full)[:, 1]
auc_full_rf = roc_auc_score(y_test, proba_full_rf)
acc_full_rf = accuracy_score(y_test, (proba_full_rf >= 0.5).astype(int))
rf_results.append(("RF - Full", auc_full_rf, acc_full_rf, X_train_full.columns, rf_full))
fpr_rf_full, tpr_rf_full, _ = roc_curve(y_test, proba_full_rf)
all_rocs.append((f"RF Full (AUC={auc_full_rf:.3f})", fpr_rf_full, tpr_rf_full))

# ----- Stepwise-selected (if present) -----
if 'X_train_step' in locals() and X_train_step.shape[1] > 0:
    rf_step, rf_bp_step, rf_cvauc_step = tune_rf_fast(X_train_step, y_train)
    proba_step_rf = rf_step.predict_proba(X_test_step)[:, 1]
    auc_step_rf = roc_auc_score(y_test, proba_step_rf)
    acc_step_rf = accuracy_score(y_test, (proba_step_rf >= 0.5).astype(int))
    rf_results.append(("RF - Stepwise", auc_step_rf, acc_step_rf, X_train_step.columns, rf_step))
    fpr_rf_step, tpr_rf_step, _ = roc_curve(y_test, proba_step_rf)
    all_rocs.append((f"RF Stepwise (AUC={auc_step_rf:.3f})", fpr_rf_step, tpr_rf_step))
else:
    print("Stepwise set is empty — skipping tuned RF for Stepwise.")

# ----------------- Print comparisons -----------------
print("\n=== Tuned Random Forest (Test Set) ===")
for name, aucv, accv, cols, model in rf_results:
    print(f"{name:18s} AUC={aucv:.3f}  ACC@0.5={accv:.3f}  (#features={len(cols)})")

# ----------------- Feature importances -----------------
def print_importances(name, model, cols, top_n=20):
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"variable": list(cols), "importance": model.feature_importances_}) \
                .sort_values("importance", ascending=False).head(top_n)
        print(f"\n{name} — Top {top_n} Feature Importances")
        print(imp.to_string(index=False))

print_importances("RF - Full", rf_results[0][4], rf_results[0][3])
if len(rf_results) > 1 and rf_results[1][0] == "RF - Stepwise":
    print_importances("RF - Stepwise", rf_results[1][4], rf_results[1][3])

# ----------------- ROC plot (RF only) -----------------
plt.figure(figsize=(7,6))
for label, fpr, tpr in all_rocs:
    plt.plot(fpr, tpr, label=label)
plt.plot([0,1],[0,1],'--',label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC — Tuned Random Forest (Test Set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------#
# Neural Network (MLP) - Renewal (yes/no)  — fast tuning + eval + ROC + permutation importances
#--------------------------------------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# ----------------- Safety checks -----------------
def _check_data(X, y, name="X"):
    X = np.asarray(X)
    y = np.asarray(y)
    if np.isnan(X).any():
        raise ValueError(f"{name} contains NaNs; impute or drop before fitting.")
    if np.unique(y).size < 2:
        raise ValueError("y has only one class; need both classes present.")

# ----------------- Fast tuner: MLP (Neural Net) -----------------
def tune_mlp_fast(X_train, y_train, random_state=42, n_iter=25, cv_splits=3, verbose=2):
    """
    Fast, small MLP tuned via RandomizedSearchCV.
    Uses StandardScaler in a Pipeline and early stopping for speed.
    """
    _check_data(X_train, y_train, "X_train (MLP)")

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            random_state=random_state,
            early_stopping=True,         # hold-out 10% of training for validation
            n_iter_no_change=10,         # patience
            max_iter=200,                # cap epochs for speed
            solver="adam",
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        ))
    ])

    # High-impact but compact search space
    param_dist = {
        "mlp__hidden_layer_sizes": [
            (16,), (32,), (64,), (32,16), (64,32)
        ],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": np.logspace(-5, -2, 6),           # L2 penalty
        "mlp__learning_rate_init": np.logspace(-4, -2, 5),
        "mlp__batch_size": [64, 128, 256, 'auto']
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=verbose,
        random_state=random_state,
        error_score="raise",
    )

    t0 = time.time()
    rs.fit(X_train, y_train)
    print(f"[tune_mlp_fast] {rs.n_iter} configs × {cv_splits}-fold in {time.time()-t0:.1f}s; best CV AUC={rs.best_score_:.4f}")
    return rs.best_estimator_, rs.best_params_, rs.best_score_

# ======================= FIT & EVALUATE =======================
# Expect these to already exist in your session:
# X_train_full, X_test_full, y_train, y_test
# Optionally: X_train_step/X_test_step, X_train_imp/X_test_imp

nn_results = []    # (name, AUC, ACC, cols, model)
nn_rocs = []       # (label, fpr, tpr)

# ----- Full features -----
nn_full, nn_bp_full, nn_cvauc_full = tune_mlp_fast(X_train_full, y_train)
proba_full_nn = nn_full.predict_proba(X_test_full)[:, 1]
auc_full_nn = roc_auc_score(y_test, proba_full_nn)
acc_full_nn = accuracy_score(y_test, (proba_full_nn >= 0.5).astype(int))
nn_results.append(("NN - Full", auc_full_nn, acc_full_nn, X_train_full.columns, nn_full))
fpr_nn_full, tpr_nn_full, _ = roc_curve(y_test, proba_full_nn)
nn_rocs.append((f"NN Full (AUC={auc_full_nn:.3f})", fpr_nn_full, tpr_nn_full))

# ----- Stepwise-selected (if present) -----
if 'X_train_step' in locals() and X_train_step.shape[1] > 0:
    nn_step, nn_bp_step, nn_cvauc_step = tune_mlp_fast(X_train_step, y_train)
    proba_step_nn = nn_step.predict_proba(X_test_step)[:, 1]
    auc_step_nn = roc_auc_score(y_test, proba_step_nn)
    acc_step_nn = accuracy_score(y_test, (proba_step_nn >= 0.5).astype(int))
    nn_results.append(("NN - Stepwise", auc_step_nn, acc_step_nn, X_train_step.columns, nn_step))
    fpr_nn_step, tpr_nn_step, _ = roc_curve(y_test, proba_step_nn)
    nn_rocs.append((f"NN Stepwise (AUC={auc_step_nn:.3f})", fpr_nn_step, tpr_nn_step))
else:
    print("Stepwise set is empty — skipping tuned NN for Stepwise.")

# ----------------- Print comparisons -----------------
print("\n=== Tuned Neural Net (MLP) — Test Set ===")
for name, aucv, accv, cols, model in nn_results:
    print(f"{name:18s} AUC={aucv:.3f}  ACC@0.5={accv:.3f}  (#features={len(cols)})")

# ----------------- Permutation importances -----------------
def print_permutation_importance(name, model, X_test, y_test, cols, top_n=20, n_repeats=10, random_state=42):
    """
    Computes permutation importance on the test set for the given pipeline model.
    Works with pipelines (scaler + mlp) since predict_proba is exposed at the pipeline level.
    """
    r = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                               n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    imp = pd.DataFrame({"variable": list(cols), "importance": r.importances_mean}) \
            .sort_values("importance", ascending=False).head(top_n)
    print(f"\n{name} — Top {top_n} Permutation Importances (test AUC impact)")
    print(imp.to_string(index=False))

# Full
print_permutation_importance("NN - Full", nn_results[0][4], X_test_full, y_test, nn_results[0][3])

# Stepwise (optional)
if len(nn_results) > 1 and nn_results[1][0] == "NN - Stepwise":
    print_permutation_importance("NN - Stepwise", nn_results[1][4], X_test_step, y_test, nn_results[1][3])

# ----------------- ROC plot (NN only) -----------------
plt.figure(figsize=(7,6))
for label, fpr, tpr in nn_rocs:
    plt.plot(fpr, tpr, label=label)
plt.plot([0,1],[0,1],'--',label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC — Tuned Neural Net (MLP) (Test Set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

