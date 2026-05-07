from typing import Dict, Any
from src.eda.schemas import EDAReport

class InsightPromptBuilder:
    @staticmethod
    def build_dataset_summary_prompt(report: EDAReport) -> str:
        """
        Build a prompt for the local LLM to generate dataset insights.
        """
        columns_info = []
        for col in report.column_profiles:
            info = f"- {col.name} ({col.type}): {col.missing_pct:.1%} missing, {col.unique_count} unique"
            if col.mean is not None:
                info += f", mean={col.mean:.2f}, std={col.std:.2f}"
            if col.skewness is not None:
                info += f", skewness={col.skewness:.2f}"
            columns_info.append(info)

        columns_str = "\n".join(columns_info)
        
        prompt = f"""
You are an expert Data Scientist and ML Engineer. 
Analyze the following dataset profile and provide high-level insights.

Dataset Name: {report.dataset_name}
Shape: {report.shape[0]} rows x {report.shape[1]} columns
Memory: {report.memory_mb:.1f} MB
Duplicate Rows: {report.duplicate_rows} ({(report.duplicate_pct * 100):.1f}%)
Overall Health Score: {report.overall_health_score:.1f}/100

Column Profiles:
{columns_str}

Detected Risks:
{", ".join(report.top_risks) if report.top_risks else "None detected by automated rules."}

Based on this data, provide:
1. A 2-sentence narrative summary of the dataset.
2. Top 3 most critical data quality or leakage risks.
3. 3 actionable preprocessing or feature engineering recommendations.
4. A brief executive summary for a non-technical stakeholder.

Format your response as a structured report.
"""
        return prompt.strip()
