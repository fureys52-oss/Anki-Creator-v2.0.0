# image_finder.py (Corrected)

import os
import re
import io
import base64
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

import fitz
import requests
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer, util

def _optimize_image(image_bytes: bytes) -> Optional[bytes]:
    try:
        # This function now only ever receives standard, Pillow-compatible formats.
        image = Image.open(io.BytesIO(image_bytes))
        
        # This logic for handling transparency and converting to RGB remains essential.
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
        # A general exception handler is still good practice.
        print(f"Error optimizing image: {e}")
        return None

def _download_image(url: str, headers: Dict[str, str]) -> Optional[bytes]:
    """
    Downloads image content from a URL and uses fitz to convert any format
    (including WebP, TIFF, SVG) into a standard PNG.
    """
    try:
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        image_bytes = response.content

        # Universal image conversion using fitz
        try:
            with fitz.open(stream=image_bytes, filetype="image") as doc:
                page = doc.load_page(0)
                pix = page.get_pixmap()
                return pix.tobytes("png")
        except Exception as e:
            print(f"Failed to process image from {url} with fitz. Error: {e}")
            return None

    except requests.RequestException as e:
        print(f"Error downloading image from {url}: {e}")
        return None
# --- The Strategy Pattern: Interfaces and Classes ---

class ImageSource(ABC):
    """Abstract base class for any image-finding strategy."""
    def __init__(self, name: str, is_enabled: bool = True):
        self.name = name
        self.is_enabled = is_enabled
        self.similarity_threshold = 0.26

    @abstractmethod
    def search(self, query_text: str, clip_model: SentenceTransformer, **kwargs) -> Optional[Dict[str, Any]]:
        pass

    def _get_image_text_similarity(self, image_bytes: bytes, text: str, clip_model: SentenceTransformer) -> float:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            image_embedding = clip_model.encode(pil_image, convert_to_tensor=True, show_progress_bar=False)
            text_embedding = clip_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
            return util.cos_sim(image_embedding, text_embedding)[0][0].item()
        except Exception as e:
            print(f"CLIP validation failed with an unexpected error: {e}")
            return 0.0

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name} (Enabled: {self.is_enabled})>"

# image_finder.py (FINAL, COMPLETE PDFImageSource Class)

class PDFImageSource(ImageSource):
    """Strategy to extract and validate images directly from a PDF."""
    def __init__(self):
        super().__init__(name="PDF (AI Validated)")

    def _is_text_heavy(self, image_bytes: bytes, threshold: int = 300) -> bool:
        """Uses OCR to check if an image is mostly text."""
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(pil_image)
            return len(text) > threshold
        except Exception:
            return False # Be safe on failure

    def _is_page_worth_rendering(self, page: fitz.Page) -> bool:
        """
        Uses a two-step heuristic to decide if a page is visually significant
        enough to be worth rendering and analyzing with CLIP.
        """
        image_area = 0.0
        for img in page.get_images(full=True):
            try:
                img_rect = page.get_image_bbox(img)
                image_area += img_rect.width * img_rect.height
            except ValueError: continue
        
        page_area = page.rect.width * page.rect.height
        if page_area > 0 and (image_area / page_area) >= 0.20:
            print(f"[{self.name}]   > Page {page.number + 1} is a candidate (high image area).")
            return True

        try:
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            if len(text := pytesseract.image_to_string(Image.open(io.BytesIO(img_bytes)))) <= 300:
                print(f"[{self.name}]   > Page {page.number + 1} is a candidate (low text density).")
                return True
            else:
                print(f"[{self.name}]   > Skipping Page {page.number + 1} (high text density, low image area).")
                return False
        except Exception:
            return True

    def search(self, query_text: str, clip_model: SentenceTransformer, **kwargs) -> Optional[Dict[str, Any]]:
        pdf_images_cache = kwargs.get("pdf_images_cache")
        source_page_numbers = kwargs.get("source_page_numbers")
        pdf_path = kwargs.get("pdf_path")

        if not pdf_path: return None

        # --- TIER 1: Search Embedded Images (Fast) ---
        if source_page_numbers and pdf_images_cache:
            relevant_images = [img for img in pdf_images_cache if img['page_num'] in source_page_numbers]
            print(f"[{self.name}] Tier 1: Searching {len(relevant_images)} embedded images from relevant pages...")
            if relevant_images:
                best_match, highest_score = self._find_best_match_in_list(relevant_images, query_text, clip_model)
                if best_match and not self._is_text_heavy(best_match["image_bytes"]):
                    print(f"[{self.name}] Tier 1 SUCCESS: Found suitable embedded image with score {highest_score:.2f}.")
                    return {"image_bytes": best_match["image_bytes"], "source": self.name, "score": highest_score}
                elif best_match:
                    print(f"[{self.name}]   > Tier 1 result was a text-heavy image. Rejecting.")

        # --- TIER 2: Page-as-Image Search (Fallback) ---
        if source_page_numbers:
            print(f"[{self.name}] Tier 1 failed. Tier 2: Filtering and analyzing entire pages as images...")
            page_render_list = []
            doc = fitz.open(pdf_path)
            for page_num in source_page_numbers:
                if page_num - 1 < len(doc):
                    page = doc.load_page(page_num - 1)
                    if self._is_page_worth_rendering(page):
                        pix = page.get_pixmap(dpi=200)
                        img_bytes = pix.tobytes("png")
                        context = page.get_text("text")
                        page_render_list.append({"image_bytes": img_bytes, "context_text": context or " "})
            
            if page_render_list:
                print(f"[{self.name}]   > Analyzing {len(page_render_list)} candidate page renders...")
                best_match, highest_score = self._find_best_match_in_list(page_render_list, query_text, clip_model)
                if best_match and not self._is_text_heavy(best_match["image_bytes"]):
                    print(f"[{self.name}] Tier 2 SUCCESS: Found suitable page render with score {highest_score:.2f}.")
                    return {"image_bytes": best_match["image_bytes"], "source": self.name, "score": highest_score}
                elif best_match:
                    print(f"[{self.name}]   > Tier 2 result was a text-heavy page render. Rejecting.")

        print(f"[{self.name}] No suitable VISUAL image found in PDF. Forcing fallback to web sources.")
        return None

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

    def _extract_images_and_context(self, pdf_path: str, pages_to_process: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        results = []
        page_iterator = [doc.load_page(i - 1) for i in pages_to_process if i - 1 < len(doc)] if pages_to_process else doc
        for page in page_iterator:
            image_list = page.get_images(full=True)
            for img_info in image_list:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_file = io.BytesIO(image_bytes)
                    img = Image.open(img_file)
                    width, height = img.size
                    if width < 35 and height < 35: continue
                    if min(width, height) > 0:
                        aspect_ratio = max(width, height) / min(width, height)
                        if aspect_ratio > 8.0: continue
                    img_rect = page.get_image_bbox(img_info)
                    context_text = self._find_context_for_image(img_rect, page)
                    if context_text:
                        results.append({"image_bytes": image_bytes, "context_text": context_text, "page_num": page.number + 1})
                except Exception as e:
                    print(f"[Image Extractor] WARNING: Could not process image xref {xref} on page {page.number + 1}. Error: {e}")
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
    def __init__(self):
        super().__init__(name="NLM Open-i")

    def search(self, query_text: str, clip_model: SentenceTransformer, **kwargs) -> Optional[Dict[str, Any]]:
        print(f"[{self.name}] Searching for: '{query_text}'")
        # --- THE CORRECTED URL and PARAMS ---
        API_URL = "https://openi.nlm.nih.gov/api/search"
        params = {"query": query_text, "it": "xg", "m": "1", "n": "5"}
        try:
            # Use `requests.get` which will handle URL encoding correctly
            response = self.session.get(API_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json().get("list", [])
            for item in data:
                image_url = f'https://openi.nlm.nih.gov{item["imgLarge"]}'
                if img_bytes := _download_image(image_url, self.session.headers):
                    score = self._get_image_text_similarity(img_bytes, query_text, clip_model)
                    if score > self.similarity_threshold:
                        print(f"[{self.name}] Found valid image with score {score:.2f}.")
                        return {"image_bytes": img_bytes, "source": self.name, "score": score}
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

    def _run_search_for_strategy(self, strategy: ImageSource, query_texts: List[str], clip_model: SentenceTransformer, pdf_path: Optional[str], page_numbers: Optional[List[int]], **kwargs) -> Dict:
        """
        A helper method to run all query variants against a single image source strategy.
        Returns the best result found by that strategy.
        """
        best_result_for_strategy = {"image_bytes": None, "score": 0.0, "source": ""}

        for query in query_texts:
            try:
                query_to_use = query
                # Use the more aggressive simplification for web sources
                if isinstance(strategy, WebImageSource):
                    image_type_words = ['diagram', 'illustration', 'chart', 'micrograph', 'photo', 'map']
                    simplified_parts = [word for word in query.split() if word.lower() not in image_type_words]
                    query_to_use = " ".join(simplified_parts[:3]) if len(simplified_parts) > 3 else " ".join(simplified_parts)
                    if query_to_use != query:
                        print(f"[{strategy.name}] Using simplified query for '{query}': '{query_to_use}'")
                
                # Call the strategy with the current query variant.
                result = strategy.search(
                    query_to_use, 
                    clip_model=clip_model, 
                    pdf_path=pdf_path, 
                    source_page_numbers=page_numbers, 
                    **kwargs
                )
                
                # If this query gives a better result for this strategy, store it.
                if result and result.get("score", 0) > best_result_for_strategy.get("score", 0):
                    best_result_for_strategy = result

            except Exception as e:
                print(f"[Error] An exception occurred in '{strategy.name}': {e}")
        
        return best_result_for_strategy

    def _finalize_image(self, result: Dict) -> Optional[str]:
        """
        A helper method to log success, optimize, and encode the winning image.
        """
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

    def find_best_image(self, query_texts: List[str], clip_model: SentenceTransformer, pdf_path: Optional[str] = None, focused_search_pages: Optional[List[int]] = None, expanded_search_pages: Optional[List[int]] = None, **kwargs) -> Optional[str]:
        
        pdf_strategy = next((s for s in self.strategies if isinstance(s, PDFImageSource)), None)

        if pdf_strategy:
            # TIER 1: FOCUSED PDF SEARCH
            if focused_search_pages:
                print(f"\n--- Trying Strategy: {pdf_strategy.name} (Focused Search on pages {focused_search_pages}) ---")
                best_result = self._run_search_for_strategy(pdf_strategy, query_texts, clip_model, pdf_path, focused_search_pages, **kwargs)
                if best_result.get("image_bytes"):
                    return self._finalize_image(best_result)

            # TIER 2: EXPANDED PDF SEARCH
            if expanded_search_pages:
                print(f"\n--- Trying Strategy: {pdf_strategy.name} (Expanded Search on pages {expanded_search_pages}) ---")
                best_result = self._run_search_for_strategy(pdf_strategy, query_texts, clip_model, pdf_path, expanded_search_pages, **kwargs)
                if best_result.get("image_bytes"):
                    return self._finalize_image(best_result)

        # TIER 3: WEB SEARCH
        web_strategies = [s for s in self.strategies if isinstance(s, WebImageSource)]
        if web_strategies:
            print("\n--- PDF Search failed or was skipped. Trying web sources... ---")
            for strategy in web_strategies:
                print(f"\n--- Trying Strategy: {strategy.name} (Web Search) ---")
                best_result = self._run_search_for_strategy(strategy, query_texts, clip_model, pdf_path, None, **kwargs)
                if best_result.get("image_bytes"):
                    return self._finalize_image(best_result)
        
        print(f"\n--- FAILED: No image found from any source for any query variant. ---")
        return None