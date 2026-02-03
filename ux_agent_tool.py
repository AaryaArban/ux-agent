from langchain.tools import BaseTool
import requests
import json

class UXAgentTool(BaseTool):
    name = "ux_agent"
    description = "Predict UX Score based on logs, behavior and text review."

    api_url: str = "http://localhost:8000/predict"

    def _run(self, logs, behavior, review_text):
        payload = {
            "logs": logs,
            "behavior": behavior,
            "review_text": review_text
        }
        response = requests.post(self.api_url, data=json.dumps(payload))
        return response.json()

    async def _arun(self, logs, behavior, review_text):
        return self._run(logs, behavior, review_text)
