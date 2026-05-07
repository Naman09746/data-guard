from typing import Dict, Any
from src.eda.schemas import EDAReport

class InsightPromptBuilder:
    @staticmethod
    def build_dataset_summary_prompt(report: EDAReport) -> str:
        """
        Build a prompt that matches the fine-tuned Lily-1.5B training format.
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
        
        # This instruction matches the generate_training_data.py exactly
        instruction = f"Analyze the following data quality profile for a dataset and provide professional recommendations."
        
        input_text = (
            f"Dataset Name: {report.dataset_name}\n"
            f"Shape: {report.shape[0]} rows x {report.shape[1]} columns\n"
            f"Memory: {report.memory_mb:.1f} MB\n"
            f"Duplicate Rows: {report.duplicate_rows} ({(report.duplicate_pct * 100):.1f}%)\n"
            f"Overall Health Score: {report.overall_health_score:.1f}/100\n\n"
            f"Column Profiles:\n{columns_str}\n\n"
            f"Detected Risks:\n{', '.join(report.top_risks) if report.top_risks else 'None detected by automated rules.'}"
        )

        # We use the Alpaca template format used during training
        prompt = (
            "Below is an instruction that describes a data quality analysis task. "
            "Write a structured professional response.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
        return prompt.strip()
