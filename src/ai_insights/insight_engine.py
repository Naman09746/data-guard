import httpx
import json
from typing import Dict, Any, Optional
from src.eda.schemas import EDAReport
from src.ai_insights.prompt_builder import InsightPromptBuilder

class InsightEngine:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "lily-1.5b"):
        self.base_url = base_url
        self.model = model
        self.prompt_builder = InsightPromptBuilder()

    async def generate_insights(self, report: EDAReport) -> Dict[str, Any]:
        """
        Generate insights using the local Ollama LLM.
        """
        prompt = self.prompt_builder.build_dataset_summary_prompt(report)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
            }
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                
                if response.status_code != 200:
                    return self._get_fallback_insights(report, f"Ollama error: {response.text}")
                
                result = response.json()
                full_text = result.get("response", "")
                
                # Simple parsing logic for the structured report
                return self._parse_llm_response(full_text)
                
        except Exception as e:
            return self._get_fallback_insights(report, str(e))

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """
        Parses the 4-part structured report from the fine-tuned model.
        """
        sections = {
            "narrative": "Insight summary unavailable.",
            "top_risks": [],
            "recommendations": [],
            "executive_summary": "Summary unavailable.",
            "raw_output": text,
            "model_name": self.model
        }

        try:
            # Split by the numbered headers we trained on
            parts = text.split("\n\n")
            
            for part in parts:
                if "1. Narrative Summary:" in part:
                    sections["narrative"] = part.replace("1. Narrative Summary:", "").strip()
                elif "2. Top Critical Risks:" in part:
                    risks = part.replace("2. Top Critical Risks:", "").strip().split("\n")
                    sections["top_risks"] = [r.strip("- ").strip("123. ") for r in risks if r.strip()]
                elif "3. Actionable Recommendations:" in part:
                    recs = part.replace("3. Actionable Recommendations:", "").strip().split("\n")
                    sections["recommendations"] = [r.strip("- ").strip("123. ") for r in recs if r.strip()]
                elif "4. Executive Summary:" in part:
                    sections["executive_summary"] = part.replace("4. Executive Summary:", "").strip()

            # Fallback if parsing failed but we have text
            if sections["narrative"] == "Insight summary unavailable." and text:
                sections["narrative"] = text[:500] + "..."

            return sections
            
        except Exception:
            return sections

    def _get_fallback_insights(self, report: EDAReport, error: str) -> Dict[str, Any]:
        return {
            "narrative": f"AI Insights temporarily unavailable. (Error: {error})",
            "top_risks": report.top_risks,
            "recommendations": report.recommendations,
            "model_name": "fallback-rules"
        }
