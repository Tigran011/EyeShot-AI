# src/ocr/ocr_engine.py

import os
import platform
import time
import io
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from PIL import Image
import importlib
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# Create our own minimal cache implementation in case the import fails
class SimpleCache:
    """Simple memory cache if the full cache implementation is unavailable"""
    def __init__(self, max_size=50):
        self._cache = {}
        self.max_size = max_size

    def generate_key(self, image, mode="auto", preprocess=True):
        """Generate a simple cache key"""
        try:
            # Create a small thumbnail for hashing
            thumb = image.copy()
            thumb.thumbnail((100, 100))
            
            # Convert to grayscale
            thumb = thumb.convert('L')
            
            # Get image bytes
            with io.BytesIO() as output:
                thumb.save(output, format='PNG')
                img_bytes = output.getvalue()
            
            # Create hash
            hash_obj = hashlib.md5(img_bytes)
            
            # Add mode and preprocess to the hash
            hash_obj.update(f"{mode}_{preprocess}".encode('utf-8'))
            
            return hash_obj.hexdigest()
        except Exception:
            return ""  # Empty string means no cache

    def get(self, key):
        """Get an item from the cache"""
        return self._cache.get(key)

    def set(self, key, value):
        """Add an item to the cache"""
        if key and value and value.get('success', False):
            self._cache[key] = value.copy()
            
            # If cache exceeds maximum size, remove oldest items
            if len(self._cache) > self.max_size:
                # Get first key (oldest)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

    def clear(self):
        """Clear the cache"""
        self._cache = {}

# Try importing from the actual cache implementation or use our simple version
try:
    from .utils.cache import OCRCache as CacheImplementation
    print("✅ Using OCRCache implementation")
except ImportError:
    try:
        from .utils.cache import ResultCache as CacheImplementation
        print("✅ Using ResultCache implementation")
    except ImportError:
        CacheImplementation = SimpleCache
        print("⚠️ Using fallback SimpleCache implementation")

# Try importing document detector implementation with different possible names
try:
    from .utils.detection import DocumentTypeDetector
    DocumentDetector = DocumentTypeDetector
    print("✅ Using DocumentTypeDetector")
except ImportError:
    try:
        from .utils.detection import DocumentDetector
        print("✅ Using DocumentDetector")
    except ImportError:
        # Simple fallback implementation
        class DocumentDetector:
            """Minimal document detector if actual implementation is unavailable"""
            def detect_document_type(self, image):
                return "standard"
        print("⚠️ Using fallback DocumentDetector implementation")

# Try importing confidence scorer from different locations
try:
    from .confidence import ConfidenceScorer
    print("✅ Using ConfidenceScorer from confidence module")
except ImportError:
    try:
        from .utils.confidence import ConfidenceScorer
        print("✅ Using ConfidenceScorer from utils.confidence module")
    except ImportError:
        # Simple fallback implementation
        class ConfidenceScorer:
            """Minimal confidence scorer if actual implementation is unavailable"""
            def calculate_confidence(self, *args, **kwargs):
                return 70.0
        print("⚠️ Using fallback ConfidenceScorer implementation")

# Import processors with fallbacks
try:
    from .image_processor import ImageProcessor
except ImportError:
    # Simple fallback implementation
    class ImageProcessor:
        """Minimal image processor if actual implementation is unavailable"""
        def preprocess_for_ocr(self, image):
            return image
    print("⚠️ Using fallback ImageProcessor implementation")

try:
    from .text_processor import TextProcessor
except ImportError:
    # Simple fallback implementation
    class TextProcessor:
        """Minimal text processor if actual implementation is unavailable"""
        def clean_extracted_text(self, text):
            return text
    print("⚠️ Using fallback TextProcessor implementation")

try:
    from .layout_analyzer import LayoutAnalyzer
except ImportError:
    # Simple fallback implementation
    class LayoutAnalyzer:
        """Minimal layout analyzer if actual implementation is unavailable"""
        def analyze_layout(self, image):
            return {}
    print("⚠️ Using fallback LayoutAnalyzer implementation")

# External OCR libraries with graceful fallbacks
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    print("✅ Tesseract engine loaded successfully")
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None
    print("❌ Tesseract not installed - OCR functionality will be limited")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("✅ EasyOCR AI engine loaded successfully")
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None
    print("⚠️ EasyOCR not installed - using Tesseract only")

try:
    import paddleocr
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
    print("✅ PaddleOCR engine loaded successfully")
except ImportError:
    PADDLE_AVAILABLE = False
    paddleocr = None
    PaddleOCR = None
    print("⚠️ PaddleOCR not installed - advanced features will be limited")

try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False
    langid = None
    print("⚠️ langid not installed - language detection disabled")

# Simple document handler for case where import fails
class SimpleDocumentHandler:
    """Minimal document handler if actual implementation is unavailable"""
    def __init__(self):
        self.processors = {}
    
    def set_processors(self, processors):
        self.processors = processors
    
    def register_engines(self, engines):
        self.engines = engines
    
    def set_languages(self, languages):
        self.languages = languages
    
    def set_quality_level(self, level):
        self.quality_level = level
    
    def set_debug_mode(self, debug_mode):
        self.debug_mode = debug_mode
    
    def set_save_debug_images(self, save_images, debug_dir='ocr_debug'):
        self.save_debug_images = save_images
        self.debug_dir = debug_dir
    
    def extract_text(self, image, preprocess=True):
        """Basic text extraction"""
        if not TESSERACT_AVAILABLE:
            return {
                'text': '',
                'confidence': 0,
                'word_count': 0,
                'char_count': 0,
                'success': False,
                'error': 'No OCR engine available'
            }
            
        try:
            # Basic extraction with tesseract
            text = pytesseract.image_to_string(image)
            conf_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Calculate confidence
            confidences = [x for x in conf_data.get('conf', []) if x > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Clean up text
            if 'text' in self.processors and hasattr(self.processors['text'], 'clean_extracted_text'):
                text = self.processors['text'].clean_extracted_text(text)
            
            # Count words and characters
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            
            return {
                'text': text,
                'confidence': avg_confidence,
                'word_count': word_count,
                'char_count': char_count,
                'success': True,
            }
        except Exception as e:
            return {
                'text': '',
                'confidence': 0,
                'word_count': 0,
                'char_count': 0,
                'success': False,
                'error': str(e)
            }


class OCREngine:
    """
    Orchestrates OCR extraction by coordinating specialized handlers and utilities.
    Manages image preprocessing, text extraction, and post-processing.
    """
    
    # Define extraction modes
    EXTRACTION_MODES = {
        'auto': 'Automatic mode selection based on content',
        'standard': 'Standard text extraction',
        'academic': 'Academic and book text with preservation of paragraphs and layout',
        'title': 'Stylized titles and headings, often with special effects or on colored backgrounds',
        'handwritten': 'Handwritten text extraction with specialized preprocessing',
        'receipt': 'Receipts and invoices with column preservation',
        'code': 'Programming code and technical text with indentation preservation',
        'table': 'Tables and structured data with cell preservation',
        'form': 'Forms with field detection',
        'mixed': 'Mixed content with multiple regions of different types',
        'id_card': 'ID cards and official documents',
        'math': 'Mathematical equations and formulas'
    }
    
    def __init__(self):
        """Initialize the OCR engine with all available backends"""
        # Fix Tesseract installation before anything else
        self._fix_tesseract_installation()
        
        # Setup paths
        self.setup_tesseract_path()
        
        # Debug settings - MOVED EARLIER so they're available when initializing handlers
        self.debug_mode = False
        self.save_debug_images = False
        self.debug_dir = 'ocr_debug'
        
        # Initialize processors
        self.image_processor = ImageProcessor()
        self.text_processor = TextProcessor()
        self.layout_analyzer = LayoutAnalyzer()
        self.document_detector = DocumentDetector()
        self.confidence_scorer = ConfidenceScorer()
        self.result_cache = CacheImplementation(max_size=50)
        
        # Initialize OCR backends
        self.easyocr_reader = None
        self.paddle_ocr = None
        self.ai_enabled = self._init_ai_engines()
        
        # Initialize handlers
        self._init_handlers()
        
        # OCR settings
        self.current_mode = 'auto'
        self.quality_level = 'balanced'  # 'speed', 'balanced', 'quality'
        self.parallel_processing = True
        self.max_workers = 4
        
        # Language settings
        self.languages = ['en']
        self.primary_language = 'en'
        self.detect_language = LANGID_AVAILABLE
        
        # Create debug directory if needed
        if self.save_debug_images and not os.path.exists(self.debug_dir):
            try:
                os.makedirs(self.debug_dir)
            except:
                self.save_debug_images = False
                
        # Register available engines with handlers
        self._register_engines_with_handlers()
        
        print("✅ OCR Engine initialized successfully")

    def _fix_tesseract_installation(self):
        """Attempt to fix Tesseract installation issues"""
        try:
            import sys
            import os
            
            # Add the src directory to the path if needed
            src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            
            # Import the fix utility
            try:
                from tesseract_fix import fix_tesseract_installation, find_tessdata_dir
            except ImportError:
                # Try alternative import paths
                try:
                    from .utils.tesseract_fix import fix_tesseract_installation, find_tessdata_dir
                except ImportError:
                    print("⚠️ Could not import tesseract_fix module")
                    return False
            
            # Try to fix the installation
            fixed = fix_tesseract_installation()
            
            if fixed:
                print("✅ Tesseract configuration fixed successfully")
                
                # Very important: Also set the tessdata path in pytesseract configuration
                import pytesseract
                tessdata_dir = find_tessdata_dir()
                if tessdata_dir:
                    # Explicitly set this for all future calls
                    os.environ["TESSDATA_PREFIX"] = os.path.normpath(tessdata_dir)
                
                # Print verification info
                print(f"✓ TESSDATA_PREFIX set to: {os.environ.get('TESSDATA_PREFIX')}")
                print(f"✓ Tesseract command path: {pytesseract.pytesseract.tesseract_cmd}")
            else:
                print("⚠️ Could not fully fix Tesseract configuration")
                print("   The application will try to use alternative OCR engines if available")
            
            return fixed
        except Exception as e:
            print(f"⚠️ Tesseract configuration error: {e}")
            print("   The application will try to use alternative OCR engines if available")
            return False 


    def _init_handlers(self):
        """Initialize all document handlers"""
        # Import handlers here to avoid circular imports
        from .handlers.base_handler import DocumentHandler
        from .handlers.academic_handler import AcademicHandler
        from .handlers.book_handler import BookHandler
        from .handlers.code_handler import CodeHandler
        from .handlers.receipt_handler import ReceiptHandler
        from .handlers.table_handler import TableHandler
        from .handlers.title_handler import TitleHandler
        from .handlers.form_handler import FormHandler
        from .handlers.list_handler import ListHandler
        from .handlers.pdf_handler import PDFHandler  # Add this import
        
        # Create handler instances
        self.handlers = {
            'academic': AcademicHandler(),
            'book': BookHandler(),
            'code': CodeHandler(),
            'receipt': ReceiptHandler(),
            'table': TableHandler(),
            'title': TitleHandler(),
            'form': FormHandler(),
            'list': ListHandler(),
            'pdf': PDFHandler(),  # Add the new PDF handler
            # Add default and specialized handlers
            'standard': DocumentHandler(),
            'handwritten': DocumentHandler(),
            'id_card': DocumentHandler(),
            'math': DocumentHandler(),
            'mixed': DocumentHandler()
        }
        
        # Map extraction modes to handler types
        self.mode_to_handler = {
            'standard': 'standard',
            'academic': 'academic',
            'book': 'book',
            'title': 'title',
            'handwritten': 'handwritten',
            'receipt': 'receipt',
            'code': 'code',
            'table': 'table',
            'form': 'form',
            'list': 'list',
            'pdf': 'pdf',  # Add this mapping
            'id_card': 'id_card',
            'math': 'math',
            'mixed': 'mixed'
        }
        
        # Register document type detection for lists
        if hasattr(self.document_detector, 'add_detector'):
            self.document_detector.add_detector(
                'list',
                lambda img: self._detect_list_document(img),
                priority=30
            )
        
        # Provide processor instances to all handlers
        for handler in self.handlers.values():
            handler.set_processors({
                'image': self.image_processor,
                'text': self.text_processor,
                'layout': self.layout_analyzer,
                'confidence': self.confidence_scorer
            })
            
            # Set debug mode
            handler.set_debug_mode(self.debug_mode)

    def _detect_list_document(self, image):
        """Detect if an image contains a list document"""
        # Basic detection: convert to text and check for bullet patterns
        try:
            # Make a quick extraction with tesseract
            if not hasattr(self, '_tesseract') or not self._tesseract:
                import pytesseract
                self._tesseract = pytesseract
                
            text = self._tesseract.image_to_string(image)
            
            # Count bullet indicators
            bullet_count = 0
            bullet_count += text.count('•')
            bullet_count += text.count('*')
            bullet_count += len(re.findall(r'^\s*-\s+', text, re.MULTILINE))
            bullet_count += len(re.findall(r'^\s*\d+\.\s+', text, re.MULTILINE))
            
            # If significant number of bullets found, it's likely a list
            if bullet_count >= 3:
                return 0.85  # High confidence
            elif bullet_count >= 1:
                return 0.6   # Medium confidence
                
            return 0.0  # Not a list
        except:
            return 0.0  # Error, not a list
    
    def _register_engines_with_handlers(self):
        """Register available OCR engines with all handlers"""
        engines = {
            'tesseract_available': TESSERACT_AVAILABLE,
            'easyocr_available': EASYOCR_AVAILABLE and self.easyocr_reader is not None,
            'paddle_available': PADDLE_AVAILABLE and self.paddle_ocr is not None
        }
        
        for handler in self.handlers.values():
            handler.register_engines(engines)
    
    def setup_tesseract_path(self):
        """Setup Tesseract executable path for multiple platforms"""
        if not TESSERACT_AVAILABLE:
            return
            
        # Common Tesseract paths by OS
        if platform.system() == "Windows":
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\%s\AppData\Local\Tesseract-OCR\tesseract.exe" % os.getenv('USERNAME', ''),
                r"C:\Users\%s\AppData\Local\Programs\Tesseract-OCR\tesseract.exe" % os.getenv('USERNAME', '')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        elif platform.system() == "Darwin":  # macOS
            # macOS homebrew and macports common paths
            possible_paths = [
                "/usr/local/bin/tesseract",
                "/opt/local/bin/tesseract",
                "/usr/bin/tesseract"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
    
    def _init_ai_engines(self) -> bool:
        """Initialize multiple AI OCR engines"""
        easyocr_initialized = False
        paddle_initialized = False
        
        # Initialize EasyOCR if available
        if EASYOCR_AVAILABLE:
            try:
                # Initialize with English as default, can be changed later
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                easyocr_initialized = True
                print("🤖 EasyOCR engine initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize EasyOCR: {e}")
        
        # Initialize PaddleOCR if available
        if PADDLE_AVAILABLE:
            try:
                # Initialize with English as default
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                paddle_initialized = True
                print("🤖 PaddleOCR engine initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize PaddleOCR: {e}")
        
        # Consider AI enabled if at least one engine is available
        return easyocr_initialized or paddle_initialized
    
    def set_languages(self, languages: List[str]) -> bool:
        """Set languages for OCR processing"""
        if not languages or not isinstance(languages, list):
            return False
            
        try:
            # Update EasyOCR
            if EASYOCR_AVAILABLE and self.easyocr_reader:
                # Reinitialize with new languages
                self.easyocr_reader = easyocr.Reader(languages, gpu=False, verbose=False)
            
            # Update PaddleOCR
            if PADDLE_AVAILABLE and self.paddle_ocr:
                # Set primary language for PaddleOCR
                primary_lang = languages[0] if languages else 'en'
                # Map to PaddleOCR's language codes if needed
                paddle_lang = primary_lang
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang=paddle_lang, show_log=False)
            
            # Store language settings
            self.languages = languages
            self.primary_language = languages[0] if languages else 'en'
            
            # Update handlers with new languages
            for handler in self.handlers.values():
                handler.set_languages(languages)
            
            return True
        except Exception as e:
            print(f"Failed to set languages: {e}")
            return False
    
    def set_extraction_mode(self, mode: str) -> bool:
        """Set the extraction mode"""
        if mode in self.EXTRACTION_MODES:
            self.current_mode = mode
            return True
        else:
            print(f"⚠️ Invalid extraction mode: {mode}. Using 'auto'.")
            self.current_mode = 'auto'
            return False
    
    def set_quality(self, level: str) -> bool:
        """Set quality level (speed vs. accuracy tradeoff)"""
        valid_levels = ['speed', 'balanced', 'quality']
        if level in valid_levels:
            self.quality_level = level
            
            # Update handler quality settings
            for handler in self.handlers.values():
                handler.set_quality_level(level)
                
            return True
        return False
    
    def extract_text(self, source: Union[str, Image.Image], mode: str = None, preprocess: bool = True) -> Dict:
        """
        Extract text from an image source using the appropriate handler
        
        Args:
            source: File path or PIL Image object
            mode: Extraction mode (auto, standard, academic, etc.)
            preprocess: Whether to apply image preprocessing
            
        Returns:
            Dictionary with extraction results
        """
        start_time = time.time()
        
        # Convert file path to PIL Image if needed
        if isinstance(source, str):
            try:
                image = Image.open(source)
            except Exception as e:
                return self._error_result(f"Failed to open image: {str(e)}")
        else:
            image = source
        
        # Check cache first with robust error handling
        cached_result = None
        try:
            mode_str = mode if mode else self.current_mode
            cache_key = self.result_cache.generate_key(image, mode_str, preprocess)
            cached_result = self.result_cache.get(cache_key) if cache_key else None
        except Exception as e:
            print(f"Cache access error (continuing without cache): {e}")
        
        if cached_result:
            # Mark as from cache and return
            cached_result['from_cache'] = True
            return cached_result
        
        # Determine extraction mode
        extraction_mode = mode if mode else self.current_mode
        
        # If auto mode, detect document type with robust error handling
        if extraction_mode == 'auto':
            try:
                extraction_mode = self.document_detector.detect_document_type(image)
            except Exception as e:
                print(f"Document type detection error (using 'standard'): {e}")
                extraction_mode = 'standard'
        
        try:
            # Select appropriate handler
            handler = self._get_handler_for_mode(extraction_mode)
            
            # Extract text using the selected handler
            result = handler.extract_text(image, preprocess)
            
            # Add processing time
            result['processing_time'] = time.time() - start_time
            
            # Add document type and mode info
            result['document_type'] = extraction_mode
            result['extraction_mode'] = extraction_mode
            
            # Cache result with robust error handling
            try:
                if hasattr(self.result_cache, 'generate_key') and hasattr(self.result_cache, 'set'):
                    cache_key = self.result_cache.generate_key(image, extraction_mode, preprocess)
                    if cache_key:
                        self.result_cache.set(cache_key, result)
            except Exception as e:
                print(f"Cache write error (continuing): {e}")
            
            return result
        
        except Exception as e:
            return self._error_result(f"Extraction error: {str(e)}")
    
    def extract_text_from_image(self, image_path: str, preprocess: bool = True) -> Dict:
        """
        Extract text from an image file with automatic mode detection
        
        Args:
            image_path: Path to image file
            preprocess: Whether to apply preprocessing
            
        Returns:
            Dictionary with extraction results
        """
        return self.extract_text(image_path, 'auto', preprocess)
    
    def extract_text_from_pil_image(self, pil_image: Image.Image, mode: str = None, preprocess: bool = True) -> Dict:
        """
        Extract text from a PIL Image with the specified mode
        
        Args:
            pil_image: PIL Image object
            mode: Extraction mode (auto, standard, academic, etc.)
            preprocess: Whether to apply preprocessing
            
        Returns:
            Dictionary with extraction results
        """
        return self.extract_text(pil_image, mode, preprocess)
    
    def _get_handler_for_mode(self, mode: str):
        """Get the appropriate handler for the specified extraction mode"""
        # Map extraction modes to handler types
        mode_to_handler = {
            'standard': 'standard',
            'academic': 'academic',
            'book': 'book',
            'title': 'title',
            'handwritten': 'handwritten',
            'receipt': 'receipt',
            'code': 'code',
            'table': 'table',
            'form': 'form',
            'id_card': 'id_card',
            'math': 'math',
            'mixed': 'mixed'
        }
        
        # Get handler type for the mode
        handler_type = mode_to_handler.get(mode, 'standard')
        
        # Return the handler (fall back to standard if not found)
        return self.handlers.get(handler_type, self.handlers['standard'])
    
    def clear_cache(self):
        """Clear the OCR result cache"""
        try:
            self.result_cache.clear()
        except Exception as e:
            print(f"Cache clear error: {e}")
    
    def _error_result(self, message: str) -> Dict:
        """Create a standardized error result"""
        return {
            'text': '',
            'confidence': 0,
            'word_count': 0,
            'char_count': 0,
            'success': False,
            'error': message,
            'timestamp': datetime.now().isoformat(),
            'processing_time': 0
        }
    
    def test_ocr_installation(self) -> Tuple[bool, str]:
        """Test if OCR engines are properly installed and working"""
        
        # Check Tesseract
        tesseract_version = None
        tesseract_working = False
        
        try:
            if TESSERACT_AVAILABLE:
                tesseract_version = pytesseract.get_tesseract_version()
                tesseract_working = tesseract_version is not None
        except Exception as e:
            print(f"Tesseract error: {e}")
            
        # Check EasyOCR
        easyocr_working = self.easyocr_reader is not None
        
        # Check PaddleOCR if applicable
        paddle_working = self.paddle_ocr is not None if hasattr(self, 'paddle_ocr') else False
        
        # Determine overall status
        is_working = tesseract_working or easyocr_working or paddle_working
        
        # Build status message
        engines = []
        if tesseract_working:
            engines.append(f"Tesseract {tesseract_version}")
        if easyocr_working:
            engines.append("EasyOCR")
        if paddle_working:
            engines.append("PaddleOCR")
            
        if is_working:
            message = f"OCR engines available: {', '.join(engines)}"
        else:
            message = "No OCR engines available. Please install Tesseract, EasyOCR, or PaddleOCR."
        
        return (is_working, message)
    
    def test_tesseract_installation(self) -> Tuple[bool, str]:
        """Compatibility method for old code that calls test_tesseract_installation()"""
        return self.test_ocr_installation()
    
    def get_engine_info(self) -> Dict:
        """Get information about available OCR engines"""
        
        tesseract_available = False
        tesseract_version = "Not available"
        
        try:
            if TESSERACT_AVAILABLE:
                version = pytesseract.get_tesseract_version()
                tesseract_available = True
                tesseract_version = str(version)
        except:
            pass
        
        return {
            'tesseract_available': tesseract_available,
            'tesseract_version': tesseract_version,
            'easyocr_available': EASYOCR_AVAILABLE and self.easyocr_reader is not None,
            'easyocr_status': 'Ready' if EASYOCR_AVAILABLE and self.easyocr_reader else 'Not available',
            'paddleocr_available': PADDLE_AVAILABLE and self.paddle_ocr is not None,
            'paddleocr_status': 'Ready' if PADDLE_AVAILABLE and self.paddle_ocr else 'Not available',
            'current_mode': self.current_mode,
            'quality_level': self.quality_level,
            'languages': self.languages,
            'primary_language': self.primary_language,
            'available_modes': list(self.EXTRACTION_MODES.keys()),
            'version': '3.0.0'
        }
        
    # Backward-compatibility methods for previous structure
    def extract_structured_pdf_text(self, pil_image: Image.Image, structure_hints: Dict = None, preprocess: bool = True) -> Dict:
        """Extract text from PDF with enhanced structure preservation"""
        try:
            # Start timing
            start_time = time.time()
            
            # Get the PDF handler specifically
            pdf_handler = self.handlers.get('pdf')
            if not pdf_handler:
                # Fall back to standard handler if PDF handler not found
                pdf_handler = self._get_handler_for_mode('standard')
            
            # Apply preprocessing if needed
            if preprocess and hasattr(self, 'image_processor'):
                processed_image = self.image_processor.preprocess_for_ocr(pil_image.copy())
            else:
                processed_image = pil_image.copy()
            
            # Use the PDF handler for extraction
            result = pdf_handler.extract_text(processed_image, False)  # Already preprocessed
            
            # Apply structure formatting if the handler has the method
            if hasattr(pdf_handler, '_apply_structure_to_ocr_text') and structure_hints:
                result['text'] = pdf_handler._apply_structure_to_ocr_text(result['text'], structure_hints)
            
            # Add processing time
            result['processing_time'] = time.time() - start_time
            
            return result
            
        except Exception as e:
            print(f"PDF extraction error: {str(e)}")
            # Create basic error result
            return {
                'text': '',
                'confidence': 0,
                'word_count': 0,
                'success': False,
                'error': f"Failed to extract structured text: {str(e)}"
            }
        
    def extract_scholarly_text(self, image: Image.Image, preprocess: bool = True) -> Dict:
        """Specialized extraction for scholarly texts"""
        return self._get_handler_for_mode('academic').extract_text(image, preprocess)
        
    def extract_book_text(self, pil_image, structure_hints=None):
        """Specialized extraction for book pages"""
        handler = self._get_handler_for_mode('book')
        result = handler.extract_text(pil_image, True)
        
        # Add structure hints if provided
        if structure_hints:
            result['structure_hints'] = structure_hints
            
        return result

    def set_debug_mode(self, debug_mode: bool):
        """Set debug mode for all components"""
        self.debug_mode = debug_mode
        
        # Update all handlers
        for handler in self.handlers.values():
            handler.set_debug_mode(debug_mode)

    def set_save_debug_images(self, save_debug_images: bool, debug_dir: str = 'ocr_debug'):
        """Configure debug image saving"""
        self.save_debug_images = save_debug_images
        self.debug_dir = debug_dir
        
        # Create directory if needed
        if self.save_debug_images and not os.path.exists(self.debug_dir):
            try:
                os.makedirs(self.debug_dir)
            except:
                self.save_debug_images = False
                print(f"Failed to create debug directory: {self.debug_dir}")
                return False
                
        # Update handlers
        for handler in self.handlers.values():
            handler.set_save_debug_images(save_debug_images, debug_dir)
            
        return True