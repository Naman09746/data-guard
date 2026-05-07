import json
import random
import os

def generate_synthetic_example():
    """Generates a single instruction-tuning pair for data quality."""
    
    # Randomly pick a domain
    domains = ["FinTech", "Healthcare", "E-commerce", "AdTech", "Logistics"]
    domain = random.choice(domains)
    
    # Randomly pick a problem
    problems = [
        "high_missingness",
        "target_leakage",
        "data_drift",
        "feature_skew",
        "duplicate_contamination",
        "healthy"
    ]
    problem = random.choice(problems)
    
    # Build statistics based on the problem
    stats = {
        "dataset_name": f"{domain.lower()}_v1.csv",
        "rows": random.randint(1000, 100000),
        "columns": random.randint(10, 50),
        "health_score": 0,
        "risks": [],
        "columns_sample": []
    }
    
    if problem == "high_missingness":
        stats["health_score"] = random.uniform(40, 60)
        stats["risks"] = ["high_missing_values", "biased_sample"]
        stats["columns_sample"].append({"name": "user_income", "missing_pct": 0.45, "unique": 1200})
    elif problem == "target_leakage":
        stats["health_score"] = random.uniform(30, 50)
        stats["risks"] = ["target_leakage_detected", "suspicious_correlation"]
        stats["columns_sample"].append({"name": "future_payment_status", "correlation": 0.99, "type": "numeric"})
    elif problem == "data_drift":
        stats["health_score"] = random.uniform(60, 80)
        stats["risks"] = ["feature_drift", "concept_drift"]
        stats["columns_sample"].append({"name": "transaction_amount", "psi_score": 0.25, "drift_detected": True})
    elif problem == "feature_skew":
        stats["health_score"] = random.uniform(70, 85)
        stats["risks"] = ["asymmetric_distribution", "outliers_detected"]
        stats["columns_sample"].append({"name": "age", "skewness": 2.4, "outliers_pct": 0.05})
    elif problem == "duplicate_contamination":
        stats["health_score"] = random.uniform(50, 70)
        stats["risks"] = ["duplicate_rows_detected", "train_test_overlap"]
        stats["columns_sample"].append({"name": "email", "duplicate_pct": 0.12})
    else:
        stats["health_score"] = random.uniform(90, 100)
        stats["risks"] = []
        stats["columns_sample"].append({"name": "id", "missing_pct": 0.0, "unique": stats["rows"]})

    # Construct the Prompt (Instruction)
    instruction = f"Analyze the following data quality profile for a {domain} dataset and provide professional recommendations."
    input_text = json.dumps(stats, indent=2)
    
    # Construct the Response (Ideal Output)
    if problem == "target_leakage":
        output = f"The {domain} dataset shows critical Target Leakage in the 'future_payment_status' column (corr=0.99). This feature likely contains information from the future that would not be available at inference time. Action: Remove this column immediately to avoid overfitting and unrealistic performance."
    elif problem == "high_missingness":
        output = f"I detected significant data gaps in the 'user_income' column (45% missing). Given this is a {domain} context, this could introduce socio-economic bias. Recommendation: Use iterative imputer or investigate if the data is Missing Not At Random (MNAR)."
    elif problem == "data_drift":
        output = f"Warning: Statistical drift detected in 'transaction_amount' (PSI=0.25). This {domain} dataset's distribution has shifted since training, which will degrade model performance. Action: Retrain the model on more recent data or implement a dynamic weighting scheme."
    elif problem == "feature_skew":
        output = f"The 'age' column shows significant skewness (2.4). In {domain} modeling, this can lead to biased predictions for certain subgroups. Recommendation: Apply a log-transform or power-transform to normalize the distribution."
    elif problem == "duplicate_contamination":
        output = f"Critical error: 12% duplicate rows detected in 'email'. This {domain} dataset likely has train-test contamination, which leads to over-optimistic performance metrics. Action: De-duplicate the dataset immediately."
    else:
        output = f"This {domain} dataset is exceptionally healthy with a score of {stats['health_score']:.1f}%. All key features show low missingness and stable distributions. It is ready for model training."

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }

def main():
    dataset = [generate_synthetic_example() for _ in range(1000)]
    output_path = os.path.join(os.path.dirname(__file__), "train_data.jsonl")
    with open(output_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    print(f"Generated {len(dataset)} synthetic training examples in {output_path}")

if __name__ == "__main__":
    main()
