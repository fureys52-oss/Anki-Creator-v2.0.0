# image_finder.py (Corrected)

import os
import re
import io
import base64
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# --- Third-party libraries ---
import fitz  # PyMuPDF
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# --- Configuration & Helpers ---
def _optimize_image(image_bytes: bytes) -> Optional[bytes]:
    """Optimizes an image by converting to JPEG and resizing if necessary."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, (0, 0), image.convert("RGBA"))
            image = background
        image = image.convert("RGB")
        if image.width > 1000 or image.height > 1000:
            image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        byte_arr = io.BytesIO()
        image.save(byte_arr, format='JPEG', quality=85, optimize=True)
        return byte_arr.getvalue()
    except Exception as e:
        print(f"Error optimizing image: {e}")
        return None

def _download_image(url: str, headers: Dict[str, str]) -> Optional[bytes]:
    """Downloads image content from a URL."""
    try:
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        print(f"Error downloading image from {url}: {e}")
        return None

# --- The Strategy Pattern: Interfaces and Classes ---

class ImageSource(ABC):
    """Abstract base class for any image-finding strategy."""
    def __init__(self, name: str, is_enabled: bool = True):
        self.name = name
        self.is_enabled = is_enabled
        self.similarity_threshold = 0.28

    @abstractmethod
    def search(self, query_text: str, clip_model: SentenceTransformer, **kwargs) -> Optional[Dict[str, Any]]:
        pass

    def _get_image_text_similarity(self, image_bytes: bytes, text: str, clip_model: SentenceTransformer) -> float:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            image_embedding = clip_model.encode(pil_image, convert_to_tensor=True, show_progress_bar=False)
            text_embedding = clip_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
            similarity_score = util.cos_sim(image_embedding, text_embedding)[0][0].item()
            return similarity_score
        except Exception as e:
            print(f"CLIP validation failed: {e}")
            return 0.0

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name} (Enabled: {self.is_enabled})>"

class PDFImageSource(ImageSource):
    """Strategy to extract and validate images directly from a PDF."""
    def __init__(self):
        super().__init__(name="PDF (AI Validated)")

    def search(self, query_text: str, clip_model: SentenceTransformer, **kwargs) -> Optional[Dict[str, Any]]:
        pdf_images_cache = kwargs.get("pdf_images_cache")
        source_page_numbers = kwargs.get("source_page_numbers")
        pdf_path = kwargs.get("pdf_path") # We need the path for rendering

        if not pdf_images_cache and not pdf_path:
            return None

        # --- TIER 1: Search Embedded Images (Fast) ---
        if source_page_numbers and pdf_images_cache:
            relevant_images = [img for img in pdf_images_cache if img['page_num'] in source_page_numbers]
            print(f"[{self.name}] Tier 1: Searching {len(relevant_images)} embedded images from relevant pages ({source_page_numbers})...")
            
            if relevant_images:
                best_match, highest_score = self._find_best_match_in_list(relevant_images, query_text, clip_model)
                if best_match:
                    print(f"[{self.name}] Tier 1 SUCCESS: Found suitable embedded image with score {highest_score:.2f}.")
                    return {"image_bytes": best_match["image_bytes"], "source": self.name, "score": highest_score}

        # --- TIER 2: Page-as-Image Search (Fallback) ---
        # This only runs if Tier 1 failed and we have page numbers to work with.
        if source_page_numbers and pdf_path:
            print(f"[{self.name}] Tier 1 failed. Tier 2: Analyzing entire pages as images...")
            
            page_render_list = []
            doc = fitz.open(pdf_path)
            for page_num in source_page_numbers:
                # Page numbers in PyMuPDF are 0-indexed
                if page_num - 1 < len(doc):
                    page = doc.load_page(page_num - 1)
                    # Render the page to a high-res image
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    # We treat the context as the full page text for better matching
                    context = page.get_text("text")
                    page_render_list.append({"image_bytes": img_bytes, "context_text": context or " "})
            
            if page_render_list:
                best_match, highest_score = self._find_best_match_in_list(page_render_list, query_text, clip_model)
                if best_match:
                    print(f"[{self.name}] Tier 2 SUCCESS: Found suitable page render with score {highest_score:.2f}.")
                    return {"image_bytes": best_match["image_bytes"], "source": self.name, "score": highest_score}

        print(f"[{self.name}] No suitable image found in PDF.")
        return None

    # --- NEW HELPER METHOD to avoid code duplication ---
    def _find_best_match_in_list(self, image_list: List[Dict], query_text: str, clip_model: SentenceTransformer) -> tuple[Optional[Dict], float]:
        best_match = None
        highest_score = 0.0

        for item in image_list:
            score_vs_query = self._get_image_text_similarity(item["image_bytes"], query_text, clip_model)
            score_vs_context = self._get_image_text_similarity(item["image_bytes"], item["context_text"], clip_model)
            final_score = max(score_vs_query, score_vs_context)

            if final_score > self.similarity_threshold and final_score > highest_score:
                highest_score = final_score
                best_match = item
        
        return best_match, highest_score

    def _extract_images_and_context(self, pdf_path: str) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        results = []
        for page in doc:
            image_list = page.get_images(full=True)
            for img_info in image_list:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # --- YOUR PREFERRED HEURISTICS ---
                    # Use Pillow to analyze the image dimensions and aspect ratio.
                    img_file = io.BytesIO(image_bytes)
                    img = Image.open(img_file)
                    width, height = img.size

                    # Rule 1: Skip if the image is too small (e.g., small icons, logos)
                    if width < 35 and height < 35:
                        continue

                    # Rule 2: Skip if the image has an extreme aspect ratio (e.g., a divider line)
                    # Avoid division by zero for single-pixel-wide/high images
                    if min(width, height) > 0:
                        aspect_ratio = max(width, height) / min(width, height)
                        if aspect_ratio > 8.0:
                            continue
                    # ---------------------------------

                    img_rect = page.get_image_bbox(img_info)
                    context_text = self._find_context_for_image(img_rect, page)
                    if context_text:
                        results.append({"image_bytes": image_bytes, "context_text": context_text, "page_num": page.number})
                except Exception as e:
                    # Log the error but continue processing other images
                    print(f"[Image Extractor] WARNING: Could not process image xref {xref} on page {page.number}. Error: {e}")
                    continue
        return results

    def _find_context_for_image(self, img_rect: fitz.Rect, page: fitz.Page) -> str:
        search_rect = img_rect + (-20, -50, 20, 20)
        text = page.get_text(clip=search_rect, sort=True)

        caption_rect = img_rect + (0, 10, 0, 50)
        caption_text = page.get_text(clip=caption_rect, sort=True)
        if re.match(r'^(Figure|Fig\.?|Table|Diagram)\s*\d+', caption_text, re.IGNORECASE):
            return caption_text.strip()

        return text.strip() if text else " "


class WebImageSource(ImageSource):
    """Base class for web-based sources with shared logic."""
    def __init__(self, name: str, api_key: Optional[str] = None, api_key_name: str = ""):
        super().__init__(name)
        self.api_key = api_key
        if not self.api_key and api_key_name:
            print(f"[{self.name}] WARNING: Env variable '{api_key_name}' not set. Disabling this source.")
            self.is_enabled = False
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'AnkiDeckGenerator/3.0 (https://github.com/your-repo)'})


class WikimediaSource(WebImageSource):
    """Strategy to find images from Wikimedia Commons."""
    # --- CORRECTED: Added __init__ method ---
    def __init__(self):
        super().__init__(name="Wikimedia")

    def search(self, query_text: str, clip_model: SentenceTransformer, **kwargs) -> Optional[Dict[str, Any]]:
        print(f"[{self.name}] Searching for: '{query_text}'")
        API_URL = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query", "format": "json", "list": "search",
            "srsearch": f"{query_text} diagram illustration", "srnamespace": "6", "srlimit": "5"
        }
        try:
            response = self.session.get(API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json().get("query", {}).get("search", [])
            if not data: return None

            image_titles = [r["title"] for r in data]
            img_params = {
                "action": "query", "format": "json", "prop": "imageinfo",
                "iiprop": "url|size", "titles": "|".join(image_titles)
            }
            img_response = self.session.get(API_URL, params=img_params, timeout=10)
            pages = img_response.json().get("query", {}).get("pages", {})

            for page in pages.values():
                if "imageinfo" in page:
                    info = page["imageinfo"][0]
                    if info["size"] > 15000:
                        if img_bytes := _download_image(info["url"], self.session.headers):
                            score = self._get_image_text_similarity(img_bytes, query_text, clip_model)
                            if score > self.similarity_threshold:
                                print(f"[{self.name}] Found valid image with score {score:.2f}.")
                                return {"image_bytes": img_bytes, "source": self.name}
        except requests.RequestException as e:
            print(f"[{self.name}] API call failed: {e}")
        return None


class NLMOpenISource(WebImageSource):
    """Strategy to find biomedical images from NLM Open-i."""
    # --- CORRECTED: Added __init__ method ---
    def __init__(self):
        super().__init__(name="NLM Open-i")

    def search(self, query_text: str, clip_model: SentenceTransformer, **kwargs) -> Optional[Dict[str, Any]]:
        print(f"[{self.name}] Searching for: '{query_text}'")
        API_URL = "https://openi.nlm.nih.gov/api/search/"
        params = {"query": query_text, "it": "xg", "m": "1", "n": "5"}
        try:
            response = self.session.get(API_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json().get("list", [])
            for item in data:
                image_url = f'https://openi.nlm.nih.gov{item["imgLarge"]}'
                if img_bytes := _download_image(image_url, self.session.headers):
                    score = self._get_image_text_similarity(img_bytes, query_text, clip_model)
                    if score > self.similarity_threshold:
                        print(f"[{self.name}] Found valid image with score {score:.2f}.")
                        return {"image_bytes": img_bytes, "source": self.name}
        except requests.RequestException as e:
            print(f"[{self.name}] API call failed: {e}")
        return None


class OpenverseSource(WebImageSource):
    """Strategy to find images from Openverse."""
    # --- CORRECTED: Added __init__ method that passes the name ---
    def __init__(self, api_key: Optional[str] = None, api_key_name: str = ""):
        super().__init__(name="Openverse", api_key=api_key, api_key_name=api_key_name)

    def search(self, query_text: str, clip_model: SentenceTransformer, **kwargs) -> Optional[Dict[str, Any]]:
        if not self.is_enabled: return None
        print(f"[{self.name}] Searching for: '{query_text}'")
        API_URL = "https://api.openverse.engineering/v1/images/"
        headers = self.session.headers.copy()
        headers['Authorization'] = f"Bearer {self.api_key}"
        params = {"q": query_text, "license_type": "all-creative-commons", "page_size": "5"}
        try:
            response = self.session.get(API_URL, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json().get("results", [])
            for item in data:
                if img_bytes := _download_image(item["url"], self.session.headers):
                    score = self._get_image_text_similarity(img_bytes, query_text, clip_model)
                    if score > self.similarity_threshold:
                        print(f"[{self.name}] Found valid image with score {score:.2f}.")
                        return {"image_bytes": img_bytes, "source": self.name}
        except requests.RequestException as e:
            print(f"[{self.name}] API call failed: {e}")
        return None


class FlickrSource(WebImageSource):
    """Strategy to find images from Flickr, filtered by license."""
    # --- CORRECTED: Added __init__ method that passes the name ---
    def __init__(self, api_key: Optional[str] = None, api_key_name: str = ""):
        super().__init__(name="Flickr", api_key=api_key, api_key_name=api_key_name)

    def search(self, query_text: str, clip_model: SentenceTransformer, **kwargs) -> Optional[Dict[str, Any]]:
        if not self.is_enabled: return None
        print(f"[{self.name}] Searching for: '{query_text}'")
        API_URL = "https://api.flickr.com/services/rest/"
        cc_licenses = "4,5,6,7,8,9,10"
        params = {
            "method": "flickr.photos.search", "api_key": self.api_key, "text": query_text,
            "format": "json", "nojsoncallback": "1", "per_page": "5", "license": cc_licenses,
            "sort": "relevance", "extras": "url_l"
        }
        try:
            response = self.session.get(API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json().get("photos", {}).get("photo", [])
            for item in data:
                if "url_l" in item:
                    if img_bytes := _download_image(item["url_l"], self.session.headers):
                        score = self._get_image_text_similarity(img_bytes, query_text, clip_model)
                        if score > self.similarity_threshold:
                            print(f"[{self.name}] Found valid image with score {score:.2f}.")
                            return {"image_bytes": img_bytes, "source": self.name}
        except requests.RequestException as e:
            print(f"[{self.name}] API call failed: {e}")
        return None


class ImageFinder:
    """Orchestrator that runs image search strategies in a prioritized order."""
    def __init__(self, strategies: List[ImageSource]):
        self.strategies = [s for s in strategies if s.is_enabled]
        print("\nImageFinder initialized with strategies:")
        for i, s in enumerate(self.strategies):
            print(f"  Priority {i+1}: {s.name}")

    def find_best_image(self, query_texts: List[str], clip_model: SentenceTransformer, pdf_path: Optional[str] = None, focused_search_pages: Optional[List[int]] = None, full_source_pages: Optional[List[int]] = None, **kwargs) -> Optional[str]:
        
        # --- TIER 1: FOCUSED PDF SEARCH ---
        # First, try the fast search on the limited, most relevant pages.
        pdf_strategy = next((s for s in self.strategies if isinstance(s, PDFImageSource)), None)

        if pdf_strategy:
            print(f"\n--- Trying Strategy: {pdf_strategy.name} (Focused Search) ---")
            
            best_result = self._run_search_for_strategy(pdf_strategy, query_texts, clip_model, pdf_path, focused_search_pages, **kwargs)
            
            if best_result.get("image_bytes"):
                return self._finalize_image(best_result) # Success! Exit early.

        # --- TIER 2: EXPANDED PDF SEARCH (FALLBACK) ---
        # This tier only runs if the focused search failed.
        if pdf_strategy and full_source_pages:
            print(f"\n--- Focused PDF search failed. Trying Expanded Search... ---")
            
            # Calculate the expanded page range (e.g., [39] -> [38, 39, 40])
            min_page = min(full_source_pages)
            max_page = max(full_source_pages)
            # Ensure we don't go below page 1
            start_page = max(1, min_page - 1)
            end_page = max_page + 1
            expanded_search_pages = list(range(start_page, end_page + 1))

            best_result = self._run_search_for_strategy(pdf_strategy, query_texts, clip_model, pdf_path, expanded_search_pages, **kwargs)

            if best_result.get("image_bytes"):
                return self._finalize_image(best_result) # Success! Exit early.

        # --- TIER 3: WEB SEARCH (FINAL FALLBACK) ---
        # This tier only runs if both PDF searches failed.
        web_strategies = [s for s in self.strategies if isinstance(s, WebImageSource)]
        for strategy in web_strategies:
            print(f"\n--- Trying Strategy: {strategy.name} (Web Search) ---")
            
            best_result = self._run_search_for_strategy(strategy, query_texts, clip_model, pdf_path, None, **kwargs)

            if best_result.get("image_bytes"):
                return self._finalize_image(best_result) # Success! Exit early.
        
        print(f"\n--- FAILED: No image found from any source for any query variant. ---")
        return None

    # --- NEW HELPER METHOD to avoid repeating code ---
    def _run_search_for_strategy(self, strategy: ImageSource, query_texts: List[str], clip_model: SentenceTransformer, pdf_path: Optional[str], page_numbers: Optional[List[int]], **kwargs) -> Dict:
        best_result_for_strategy = {"image_bytes": None, "score": 0.0}

        for query in query_texts:
            query_to_use = query
            if isinstance(strategy, WebImageSource):
                image_type_words = ['diagram', 'illustration', 'chart', 'micrograph', 'photo', 'map']
                simplified_parts = [word for word in query.split() if word.lower() not in image_type_words]
                query_to_use = " ".join(simplified_parts[:3]) if len(simplified_parts) > 3 else " ".join(simplified_parts)
                if query_to_use != query:
                    print(f"[{strategy.name}] Using simplified query for '{query}': '{query_to_use}'")

            result = strategy.search(query_to_use, clip_model=clip_model, pdf_path=pdf_path, source_page_numbers=page_numbers, **kwargs)
            
            if result and result.get("score", 0) > best_result_for_strategy.get("score", 0):
                best_result_for_strategy = result
        
        return best_result_for_strategy

    # --- NEW HELPER METHOD to finalize the image ---
    def _finalize_image(self, result: Dict) -> Optional[str]:
        score = result['score']
        source = result['source']
        image_bytes = result['image_bytes']
        
        print(f"--- SUCCESS: Found suitable image via {source} with score {score:.2f}. Halting search for this card. ---")

        if optimized_bytes := _optimize_image(image_bytes):
            b64_image = base64.b64encode(optimized_bytes).decode('utf-8')
            return f'<img src="data:image/jpeg;base64,{b64_image}">'
        else:
            print(f"--- WARNING: Found image via {source}, but failed to optimize it. ---")
            return None