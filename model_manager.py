# model_manager.py (Corrected)

import requests
from typing import List, Dict, Any, Optional

# Updated based on the user-provided screenshot of current free-tier models.
KNOWN_MODELS = [
    # Pro Models (for high-quality generation)
    {"name": "gemini-2.5-pro", "role": "pro", "quality_rank": 5},
    {"name": "gemini-1.5-pro", "role": "pro", "quality_rank": 4}, # Kept for future-proofing

    # Flash Models (for high-volume extraction)
    # The model manager sorts by RPM, then RPD.
    # gemini-2.0-flash-lite (30 RPM) is not on the old list and is a great addition.
    {"name": "gemini-2.0-flash-lite", "role": "flash", "rpm": 30, "rpd": 200},
    {"name": "gemini-2.5-flash-lite", "role": "flash", "rpm": 15, "rpd": 1000},
    {"name": "gemini-2.0-flash", "role": "flash", "rpm": 15, "rpd": 200}, # Renamed from gemini-2.0-flash-001 for clarity
    {"name": "gemini-2.5-flash", "role": "flash", "rpm": 10, "rpd": 250},
]

class GeminiModelManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.available_models = self._list_available_models()

    def _list_available_models(self) -> List[str]:
        """Queries the Google API to get a list of currently available models."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
        try:
            response = requests.get(url, timeout=15)
            # Add robust error handling for common issues like invalid keys or disabled APIs
            if response.status_code == 400:
                 print("[Model Manager] CRITICAL: API call failed with 'Bad Request'. Your API Key is likely invalid.")
                 return []
            if response.status_code == 403:
                print("[Model Manager] CRITICAL: API call failed with 'Permission Denied'. Ensure the 'Generative Language API' is enabled in your Google Cloud project.")
                return []
            response.raise_for_status()
            data = response.json()
            return [model['name'].replace('models/', '') for model in data.get('models', [])]
        except requests.RequestException as e:
            print(f"[Model Manager] CRITICAL: Could not connect to Google API: {e}")
            return []

    def get_optimal_models(self) -> Optional[Dict[str, Any]]:
        """Selects the best available Pro and Flash models based on our priority list."""
        if not self.available_models:
            print("[Model Manager] No available models found. Cannot select optimal configuration.")
            return None

        print("\n--- Model Discovery ---")
        print(f"Found {len(self.available_models)} available models from API.")

        # Find the best available "Pro" model
        available_pro = sorted([m for m in KNOWN_MODELS if m['role'] == 'pro' and m['name'] in self.available_models], key=lambda x: x['quality_rank'], reverse=True)
        if not available_pro:
            print("[Model Manager] CRITICAL: No known 'Pro' model is available via your API key.")
            return None
        best_pro_model = available_pro[0]

        # Find the best available "Flash" model
        available_flash = sorted([m for m in KNOWN_MODELS if m['role'] == 'flash' and m['name'] in self.available_models], key=lambda x: (x.get('rpm', 0), x.get('rpd', 0)), reverse=True)
        if not available_flash:
            print("[Model Manager] CRITICAL: No known 'Flash' model is available. Extraction will fail.")
            return None
        best_flash_model = available_flash[0]

        result = {
            "pro_model_name": best_pro_model['name'],
            "flash_model_name": best_flash_model['name'],
            "flash_model_rpm": best_flash_model.get('rpm', 15) # Default RPM
        }
        
        print(f"Selected PRO model for card generation: {result['pro_model_name']}")
        print(f"Selected FLASH model for fact extraction: {result['flash_model_name']} (RPM Limit: {result['flash_model_rpm']})")
        print("-----------------------\n")
        return result