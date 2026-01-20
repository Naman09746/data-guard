# Churn Dataset Case Study

This case study demonstrates the full capabilities of the Data Quality & Leakage Detection System using a realistic customer churn prediction scenario.

## Overview

We generate a synthetic telecom customer dataset with:
- **Clean features**: Customer demographics and usage patterns
- **Leaky features**: Information that shouldn't be available at prediction time

The system then:
1. Detects the leaky features
2. Trains a model WITH leakage (inflated performance)
3. Removes leaky features and retrains (realistic performance)
4. Shows the business impact of leakage

## Running the Case Study

```bash
cd case_studies/churn_dataset
python run_case_study.py
```

## Generated Files

After running, you'll have:

| File | Description |
|------|-------------|
| `raw_data.csv` | Synthetic churn dataset with 5,000 samples |
| `leakage_found.md` | Detailed leakage detection report |
| `metrics_before.md` | Model metrics WITH leakage (inflated) |
| `metrics_after.md` | Model metrics AFTER cleanup (realistic) |

## Expected Results

### Leaky Features Detected

| Feature | Why It's Leaky |
|---------|----------------|
| `churn_risk_score` | Directly derived from target |
| `cancellation_survey_score` | Only collected after churn |
| `final_bill_amount` | Calculated based on churn decision |

### Performance Comparison

| Metric | With Leakage | After Cleanup |
|--------|--------------|---------------|
| Accuracy | ~95%+ | ~75-80% |
| F1 Score | ~94%+ | ~70-75% |

The "drop" in performance is actually a **correction** — the clean metrics reflect what the model will achieve in production.

## Key Takeaways

1. **Leakage causes unrealistic expectations** - Models appear to perform much better than they actually will
2. **ML-based detection catches subtle leaks** - Rule-based approaches might miss some of these
3. **Impact experiments prove business value** - Shows stakeholders why leakage matters
4. **Documentation is essential** - This case study serves as a reference for future projects
