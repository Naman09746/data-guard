"""
DataGuard AI - Synthetic Training Data Generator

Generates instruction-tuning pairs that EXACTLY match the format used by
src/ai_insights/prompt_builder.py at inference time. This ensures the
fine-tuned model produces well-structured, actionable insights.
"""

import json
import os
import random
from typing import Any

# ─── Constants ────────────────────────────────────────────────────────────────

DOMAINS = ["FinTech", "Healthcare", "E-commerce", "AdTech", "Logistics"]

COLUMN_POOLS = {
    "FinTech":      ["transaction_amount", "account_age_days", "credit_score", "loan_to_value", "fraud_label", "income_bucket"],
    "Healthcare":   ["patient_age", "bmi", "blood_pressure", "readmission_flag", "diagnosis_code", "insurance_type"],
    "E-commerce":   ["purchase_value", "session_duration", "cart_abandonment", "return_rate", "customer_ltv", "promo_code_used"],
    "AdTech":       ["ctr", "bid_price", "impression_count", "conversion_rate", "user_engagement_score", "ad_spend"],
    "Logistics":    ["delivery_time_hrs", "distance_km", "package_weight_kg", "route_risk_score", "carrier_delay_flag", "on_time_flag"],
}

PROBLEM_TYPES = [
    "high_missingness",
    "target_leakage",
    "data_drift",
    "feature_skew",
    "duplicate_contamination",
    "healthy",
    "multi_issue",  # Combined scenario for robustness
]

# ─── Stats Builders ────────────────────────────────────────────────────────────

def _build_column_profile(domain: str, problem: str) -> list[dict[str, Any]]:
    """Generate realistic column profiles for the given domain/problem."""
    cols = random.sample(COLUMN_POOLS[domain], k=min(4, len(COLUMN_POOLS[domain])))
    profiles = []
    for col in cols:
        profile: dict[str, Any] = {
            "name": col,
            "type": random.choice(["numeric", "numeric", "categorical"]),
            "missing_pct": round(random.uniform(0.0, 0.05), 3),
            "unique": random.randint(10, 5000),
        }
        # Layer problem-specific signals onto random columns
        if problem == "high_missingness" and col == cols[0]:
            profile["missing_pct"] = round(random.uniform(0.30, 0.60), 3)
        if problem == "target_leakage" and col == cols[0]:
            profile["correlation_with_target"] = round(random.uniform(0.95, 0.99), 3)
        if problem == "data_drift" and col == cols[0]:
            profile["psi_score"] = round(random.uniform(0.22, 0.45), 3)
            profile["drift_detected"] = True
        if problem == "feature_skew" and col == cols[0]:
            profile["skewness"] = round(random.uniform(2.0, 5.5), 2)
        if problem == "duplicate_contamination" and col == cols[0]:
            profile["duplicate_pct"] = round(random.uniform(0.08, 0.20), 3)
        if profile["type"] == "numeric":
            profile["mean"] = round(random.uniform(1, 1000), 2)
            profile["std"] = round(random.uniform(0.1, 300), 2)

        profiles.append(profile)
    return profiles


def build_stats(domain: str, problem: str) -> dict[str, Any]:
    """Build a realistic dataset stat block matching the InsightEngine prompt format."""
    rows = random.randint(2000, 150000)
    cols = random.randint(8, 60)
    duplicate_pct = round(random.uniform(0.01, 0.03), 3)
    health_score = 0.0

    risks: list[str] = []
    if problem == "high_missingness":
        health_score = round(random.uniform(40, 62), 1)
        risks = ["high_missing_values", "biased_sample"]
    elif problem == "target_leakage":
        health_score = round(random.uniform(28, 52), 1)
        risks = ["target_leakage_detected", "suspicious_correlation"]
    elif problem == "data_drift":
        health_score = round(random.uniform(58, 75), 1)
        risks = ["feature_drift", "distribution_shift"]
    elif problem == "feature_skew":
        health_score = round(random.uniform(68, 82), 1)
        risks = ["asymmetric_distribution", "outliers_detected"]
    elif problem == "duplicate_contamination":
        health_score = round(random.uniform(45, 68), 1)
        risks = ["duplicate_rows_detected", "train_test_contamination"]
        duplicate_pct = round(random.uniform(0.10, 0.22), 3)
    elif problem == "multi_issue":
        health_score = round(random.uniform(20, 45), 1)
        risks = random.sample(
            ["high_missing_values", "feature_drift", "outliers_detected",
             "target_leakage_detected", "duplicate_rows_detected"], k=3
        )
    else:  # healthy
        health_score = round(random.uniform(88, 100), 1)
        risks = []

    return {
        "dataset_name": f"{domain.lower()}_{random.randint(1, 5)}_{random.choice(['train','prod','batch'])}.csv",
        "shape": [rows, cols],
        "memory_mb": round(rows * cols * 8 / 1_000_000, 2),
        "duplicate_rows": int(rows * duplicate_pct),
        "duplicate_pct": duplicate_pct,
        "overall_health_score": health_score,
        "top_risks": risks,
        "column_profiles": _build_column_profile(domain, problem),
    }


# ─── Output Builder ────────────────────────────────────────────────────────────

def build_output(domain: str, problem: str, stats: dict[str, Any]) -> str:
    """
    Build a structured response that matches the 4-part format requested
    by prompt_builder.py at inference time.
    """
    score = stats["overall_health_score"]
    risks = stats["top_risks"]
    col = stats["column_profiles"][0]
    col_name = col["name"]
    rows = stats["shape"][0]

    # 1. Narrative summary
    if problem == "high_missingness":
        mp = col.get("missing_pct", 0.45)
        narrative = (
            f"The {domain} dataset ({rows:,} rows) shows a critical missingness pattern "
            f"in '{col_name}' ({mp:.0%} missing), which is likely to introduce significant bias. "
            f"The overall health score of {score:.1f}/100 signals the dataset requires cleaning before training."
        )
        risks_str = "1. High missingness in key feature, risk of biased model predictions.\n2. Missing Not At Random (MNAR) pattern could skew learned distributions."
        recommendations = (
            "1. Apply IterativeImputer or MissForest to impute '{col_name}' using correlated features.\n"
            "2. Add a binary missingness indicator column to preserve the signal.\n"
            "3. Investigate the root cause of missing data — if MNAR, imputation alone is insufficient."
        ).format(col_name=col_name)
        executive = f"This {domain} dataset has data gaps that need to be addressed before building a reliable model. Your data engineering team should investigate and repair the '{col_name}' column."

    elif problem == "target_leakage":
        corr = col.get("correlation_with_target", 0.98)
        narrative = (
            f"Critical target leakage detected in the {domain} dataset. '{col_name}' has a correlation of "
            f"{corr:.2f} with the target variable, strongly indicating future data contamination. "
            f"Health score: {score:.1f}/100 — this dataset will produce misleading model performance."
        )
        risks_str = "1. Target leakage will inflate training accuracy by up to 30-40%.\n2. The model will fail catastrophically in production where this future data is unavailable."
        recommendations = (
            f"1. Remove '{col_name}' from the feature set immediately.\n"
            "2. Audit all features with target correlation > 0.85 for similar leakage.\n"
            "3. Implement a strict temporal cutoff to prevent future data from entering training."
        )
        executive = f"This {domain} dataset has a hidden flaw: one feature is leaking information from the future into the model. This will cause the AI to look highly accurate in testing but fail in real usage."

    elif problem == "data_drift":
        psi = col.get("psi_score", 0.28)
        narrative = (
            f"Significant statistical drift detected in the {domain} dataset. '{col_name}' has a PSI score of "
            f"{psi:.3f} (threshold: 0.2), indicating the production distribution has diverged from training. "
            f"Health score: {score:.1f}/100."
        )
        risks_str = "1. Model predictions will degrade silently as the data distribution continues to shift.\n2. PSI > 0.25 is a critical threshold requiring immediate model retraining."
        recommendations = (
            f"1. Retrain the model using a data window that includes recent '{col_name}' distributions.\n"
            "2. Implement automated PSI monitoring in the CI/CD pipeline with alerting.\n"
            "3. Apply importance weighting to recent samples during training to reduce drift sensitivity."
        )
        executive = f"The {domain} AI model's training data no longer reflects current real-world conditions. Predictions are likely becoming less accurate over time and a model refresh is needed."

    elif problem == "feature_skew":
        skew = col.get("skewness", 2.8)
        narrative = (
            f"The {domain} dataset exhibits high feature skewness in '{col_name}' (skewness={skew:.2f}), "
            f"which will negatively impact linear models and distance-based algorithms. "
            f"Health score: {score:.1f}/100."
        )
        risks_str = "1. Heavily skewed features reduce gradient stability during training.\n2. Outliers in skewed distributions can dominate loss computation and distort learned weights."
        recommendations = (
            f"1. Apply log1p or Box-Cox transformation to '{col_name}' to normalize the distribution.\n"
            "2. Use RobustScaler instead of StandardScaler to reduce outlier sensitivity.\n"
            "3. Consider winsorizing extreme values (clip at 1st/99th percentile) before transformation."
        )
        executive = f"Some data fields in the {domain} dataset have extreme value distributions that can make the AI model less reliable. Simple data preprocessing steps can resolve this issue."

    elif problem == "duplicate_contamination":
        dup_pct = stats["duplicate_pct"]
        narrative = (
            f"The {domain} dataset has {dup_pct:.0%} duplicate rows ({stats['duplicate_rows']:,} records), "
            f"creating a high risk of train-test contamination if not removed before splitting. "
            f"Health score: {score:.1f}/100."
        )
        risks_str = "1. Duplicate rows in training data will cause the model to memorize examples rather than generalize.\n2. If duplicates leak into the test set, performance metrics will be over-optimistic by design."
        recommendations = (
            "1. Deduplicate using all feature columns before performing the train-test split.\n"
            "2. If row-level duplicates are intentional (e.g., repeated transactions), use a groupby-based split.\n"
            "3. Audit the data ingestion pipeline to prevent duplicate writes at the source."
        )
        executive = f"The {domain} dataset contains repeated data entries that can give a false impression of how well the AI model works. Removing duplicates is a quick fix that should be applied immediately."

    elif problem == "multi_issue":
        narrative = (
            f"The {domain} dataset (health score: {score:.1f}/100) presents multiple simultaneous data quality issues "
            f"including {', '.join(risks[:2])}. This combination severely limits the dataset's readiness for model training."
        )
        risks_str = "\n".join(f"{i+1}. {r.replace('_', ' ').title()}." for i, r in enumerate(risks))
        recommendations = (
            "1. Address data missingness first, as it affects all downstream quality checks.\n"
            "2. After imputation, re-run a full leakage and drift audit to get accurate secondary diagnostics.\n"
            "3. Establish a data validation gate in your MLOps pipeline to prevent multi-issue datasets from reaching training."
        )
        executive = f"The {domain} dataset has multiple data problems that need to be fixed in sequence. This requires a structured data cleaning sprint before any model development can proceed."

    else:  # healthy
        narrative = (
            f"The {domain} dataset ({rows:,} rows, {score:.1f}/100 health score) is in excellent condition. "
            f"Key features show low missingness, stable distributions, and no leakage signals. "
            f"The dataset is ready for model training."
        )
        risks_str = "No critical risks detected by automated rules."
        recommendations = (
            "1. Proceed with model training using the current feature set.\n"
            "2. Establish a baseline PSI monitoring dashboard to detect future drift in production.\n"
            "3. Document current data distributions as the reference baseline for future audits."
        )
        executive = f"The {domain} dataset is clean, complete, and ready to use. The AI team can proceed with model development with high confidence."

    return (
        f"1. Narrative Summary:\n{narrative}\n\n"
        f"2. Top Critical Risks:\n{risks_str}\n\n"
        f"3. Actionable Recommendations:\n{recommendations}\n\n"
        f"4. Executive Summary:\n{executive}"
    )


# ─── Core Generator ────────────────────────────────────────────────────────────

def generate_synthetic_example() -> dict[str, Any]:
    """Generates a single instruction-tuning pair aligned with inference format."""
    domain = random.choice(DOMAINS)
    problem = random.choice(PROBLEM_TYPES)
    stats = build_stats(domain, problem)

    # Build the prompt in the SAME format as prompt_builder.py
    col_lines = []
    for col in stats["column_profiles"]:
        line = f"- {col['name']} ({col['type']}): {col['missing_pct']:.1%} missing, {col['unique']} unique"
        if "mean" in col:
            line += f", mean={col['mean']:.2f}, std={col['std']:.2f}"
        if "skewness" in col:
            line += f", skewness={col['skewness']:.2f}"
        col_lines.append(line)

    instruction = f"Analyze the following data quality profile for a {domain} dataset and provide professional recommendations."
    input_text = (
        f"Dataset Name: {stats['dataset_name']}\n"
        f"Shape: {stats['shape'][0]} rows x {stats['shape'][1]} columns\n"
        f"Memory: {stats['memory_mb']:.1f} MB\n"
        f"Duplicate Rows: {stats['duplicate_rows']} ({stats['duplicate_pct'] * 100:.1f}%)\n"
        f"Overall Health Score: {stats['overall_health_score']:.1f}/100\n\n"
        f"Column Profiles:\n" + "\n".join(col_lines) + "\n\n"
        f"Detected Risks:\n{', '.join(stats['top_risks']) if stats['top_risks'] else 'None detected by automated rules.'}"
    )
    output = build_output(domain, problem, stats)

    return {"instruction": instruction, "input": input_text, "output": output}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    n_examples = 3000  # Increased from 1000 for better generalization
    dataset = [generate_synthetic_example() for _ in range(n_examples)]
    output_path = os.path.join(os.path.dirname(__file__), "train_data.jsonl")
    with open(output_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    print(f"Generated {len(dataset)} training examples → {output_path}")


if __name__ == "__main__":
    main()
