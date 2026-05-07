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
        # This is a simplified parser. In a real scenario, we might use regex or 
        # ask the LLM for JSON directly.
        return {
            "narrative": text.split("1.")[0].strip() if "1." in text else text,
            "raw_output": text,
            "model_name": self.model
        }

    def _get_fallback_insights(self, report: EDAReport, error: str) -> Dict[str, Any]:
        return {
            "narrative": f"AI Insights temporarily unavailable. (Error: {error})",
            "top_risks": report.top_risks,
            "recommendations": report.recommendations,
            "model_name": "fallback-rules"
        }
