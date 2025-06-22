"""
EyeShot AI - Complete Document Handler with Full Integration
Author: Tigran0000
Last updated: 2025-06-22 15:21:05 UTC
"""

import os
import re
import logging
import importlib
import inspect
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from collections import defaultdict

# Third-party imports with graceful fallbacks
try:
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtCore import QSize
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False
    class QPixmap: pass
    class QSize: pass

# Image processing
from PIL import Image

# Optional dependencies with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Configure logging
logger = logging.getLogger("EyeShot.FileHandler")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class FileHandler:
    """Complete document handler with all handler types integrated"""
    
    # Supported file formats
    SUPPORTED_IMAGES = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    SUPPORTED_PDFS = {'.pdf'}
    SUPPORTED_ALL = SUPPORTED_IMAGES | SUPPORTED_PDFS
    
    def __init__(self, user_id: str = None):
        """Initialize with full handler integration"""
        self.current_file = None
        self.file_type = None
        self.current_user = user_id or "Tigran0000"
        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.debug_mode = False
        self.log_details = True
        
        # Initialize all available handlers
        self.handlers = self._initialize_handlers()
        
        # List of all handler types we should support
        self.supported_handler_types = [
            'academic', 'book', 'code', 'form', 'list', 
            'pdf', 'receipt', 'table', 'title'
        ]
        
        # Verify all handlers are loaded
        self._verify_handlers()
        
        logger.info(f"🚀 EyeShot FileHandler initialized")
        logger.info(f"👤 User: {self.current_user}")
        logger.info(f"⏰ Time: {self.current_time}")
        logger.info(f"🔧 Handlers registered: {', '.join(self.handlers.keys())}")
        
    def _initialize_handlers(self) -> Dict[str, Any]:
        """Load and initialize all available document handlers"""
        handlers = {}
        
        try:
            # Find handlers directory
            handlers_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ocr', 'handlers')
            
            if os.path.exists(handlers_dir):
                # Get all handler files
                handler_files = [f for f in os.listdir(handlers_dir) 
                               if f.endswith('_handler.py') and f != 'base_handler.py']
                
                # Load each handler
                for handler_file in handler_files:
                    try:
                        # Get handler name and type
                        handler_name = handler_file[:-3]  # Remove .py
                        handler_type = handler_name.replace('_handler', '')
                        
                        # Import and initialize the handler
                        module_path = f"ocr.handlers.{handler_name}"
                        module = importlib.import_module(module_path)
                        
                        # Find handler class in module
                        for class_name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and class_name.endswith('Handler') and class_name != 'DocumentHandler':
                                handlers[handler_type] = obj()
                                break
                    except Exception as e:
                        logger.warning(f"Failed to load handler '{handler_file}': {e}")
            
            # If auto-discovery failed, fall back to manual imports
            if not handlers:
                logger.warning("Handler auto-discovery failed, using manual imports")
                handlers = self._load_handlers_manually()
                
            # Ensure we have a book handler as fallback
            if 'book' not in handlers and handlers:
                logger.warning("BookHandler not found, using alternative as fallback")
                first_handler = next(iter(handlers.values()))
                handlers['book'] = first_handler
                
            return handlers
                
        except Exception as e:
            logger.error(f"Error initializing handlers: {e}")
            return self._load_handlers_manually()
    
    def _load_handlers_manually(self) -> Dict[str, Any]:
        """Manual fallback for loading handlers - includes ALL handlers"""
        handlers = {}
        
        # Load BookHandler first (most important fallback)
        try:
            from ocr.handlers.book_handler import BookHandler
            handlers['book'] = BookHandler()
        except Exception as e:
            logger.error(f"Failed to load BookHandler: {e}")
        
        # Try to import all other handlers
        handler_classes = [
            ('academic', 'AcademicHandler'),
            ('code', 'CodeHandler'),
            ('form', 'FormHandler'),
            ('list', 'ListHandler'),
            ('pdf', 'PdfHandler'),
            ('receipt', 'ReceiptHandler'),
            ('table', 'TableHandler'),
            ('title', 'TitleHandler')  # Added title handler
        ]
        
        for handler_type, class_name in handler_classes:
            try:
                module = importlib.import_module(f"ocr.handlers.{handler_type}_handler")
                handler_class = getattr(module, class_name)
                handlers[handler_type] = handler_class()
                logger.info(f"✓ Loaded {class_name} manually")
            except Exception as e:
                logger.warning(f"Failed to load {class_name} manually: {e}")
                
        return handlers
    
    def _verify_handlers(self):
        """Verify all expected handlers are loaded and log any missing ones"""
        missing_handlers = []
        
        for handler_type in self.supported_handler_types:
            if handler_type not in self.handlers:
                missing_handlers.append(handler_type)
                
        if missing_handlers:
            logger.warning(f"⚠️ Missing handlers: {', '.join(missing_handlers)}")
        else:
            logger.info("✓ All expected handlers are loaded")
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate file exists and is supported"""
        
        # Check if file exists
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Check if it's a file (not directory)
        if not os.path.isfile(file_path):
            return False, "Path is not a file"
        
        # Check file size (optional - prevent huge files)
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return False, "File is too large (max 100MB)"
        
        # Check if format is supported
        if not self.is_supported_file(file_path):
            extension = Path(file_path).suffix.lower()
            supported = ", ".join(sorted(self.SUPPORTED_ALL))
            return False, f"Unsupported format '{extension}'. Supported formats: {supported}"
        
        return True, "File is valid"
    
    def is_supported_file(self, file_path: str) -> bool:
        """Check if file format is supported"""
        if not file_path:
            return False
        
        extension = Path(file_path).suffix.lower()
        return extension in self.SUPPORTED_ALL
    
    def get_file_type(self, file_path: str) -> Optional[str]:
        """Determine file type (image or pdf)"""
        if not file_path:
            return None
        
        extension = Path(file_path).suffix.lower()
        
        if extension in self.SUPPORTED_IMAGES:
            return 'image'
        elif extension in self.SUPPORTED_PDFS:
            return 'pdf'
        else:
            return None
    
    def get_file_info(self, file_path: str) -> dict:
        """Get detailed file information"""
        
        if not os.path.exists(file_path):
            return {}
        
        file_stat = os.stat(file_path)
        file_path_obj = Path(file_path)
        
        return {
            'name': file_path_obj.name,
            'path': file_path,
            'size': file_stat.st_size,
            'size_mb': round(file_stat.st_size / (1024 * 1024), 2),
            'extension': file_path_obj.suffix.lower(),
            'type': self.get_file_type(file_path),
            'modified': file_stat.st_mtime
        }

    def load_image_as_pixmap(self, image_path: str, max_size: QSize = None) -> Optional[QPixmap]:
        """Load image file as QPixmap with optional size limit"""
        
        if not QT_AVAILABLE:
            return None
            
        try:
            pixmap = QPixmap(image_path)
            
            if pixmap.isNull():
                return None
            
            # Scale down if too large (for display performance)
            if max_size and (pixmap.width() > max_size.width() or pixmap.height() > max_size.height()):
                pixmap = pixmap.scaled(max_size, 1, 1)  # Keep aspect ratio, smooth transform
            
            return pixmap
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None      
            
    def get_file_filter_string(self) -> str:
        """Get file filter string for QFileDialog"""
        
        return (
            "All Supported (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.pdf);;"
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;"
            "PDF Files (*.pdf);;"
            "All Files (*)"
        )
    
    def _detect_list_document(self, image) -> float:
        """Detect if an image contains a list document"""
        if not TESSERACT_AVAILABLE:
            return 0.0
            
        try:
            # Create a smaller version for quick analysis
            small_img = image.copy()
            small_img.thumbnail((800, 600), Image.LANCZOS)
            
            # Quick OCR with simple config
            text = pytesseract.image_to_string(small_img)
            
            # Count bullet indicators
            bullet_count = 0
            bullet_count += text.count('•')
            bullet_count += text.count('*')
            bullet_count += len(re.findall(r'^\s*-\s+', text, re.MULTILINE))
            bullet_count += len(re.findall(r'^\s*\d+\.\s+', text, re.MULTILINE))
            bullet_count += len(re.findall(r'^\s*[a-z]\)\s+', text, re.MULTILINE))
            
            # If significant number of bullets found, it's likely a list
            if bullet_count >= 5:
                # High confidence
                return 0.9
            elif bullet_count >= 3:
                # Medium-high confidence
                return 0.8
            elif bullet_count >= 1:
                # Medium confidence
                return 0.6
                
            return 0.0  # Not a list
        except Exception as e:
            logger.error(f"Error in list detection: {e}")
            return 0.0  # Error, not a list
    
    def _extract_visual_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract visual features from the image"""
        features = {}
        
        try:
            # Basic image properties
            width, height = image.size
            features['width'] = width
            features['height'] = height
            features['aspect_ratio'] = width / height
            
            # Classify aspect ratio
            features['is_portrait'] = features['aspect_ratio'] < 1.0
            features['is_landscape'] = features['aspect_ratio'] > 1.0
            features['is_square'] = 0.95 <= features['aspect_ratio'] <= 1.05
            features['is_narrow'] = features['aspect_ratio'] < 0.5 or features['aspect_ratio'] > 2.0
            features['is_standard_paper'] = 0.65 <= features['aspect_ratio'] <= 0.75  # Standard paper ratio
            
            # Process more detailed features if numpy is available
            if NUMPY_AVAILABLE:
                # Calculate text density and layout features
                features.update(self._analyze_layout(image))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return {'width': 0, 'height': 0, 'aspect_ratio': 1.0}
    
    def _analyze_layout(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze document layout patterns"""
        features = {}
        
        try:
            # Convert to greyscale and numpy array
            gray_img = image.convert('L')
            img_array = np.array(gray_img)
            
            # Detect text density
            text_threshold = 200  # Darker pixels are likely text
            text_mask = img_array < text_threshold
            text_ratio = np.sum(text_mask) / text_mask.size
            features['text_density'] = float(text_ratio)
            
            # Detect horizontal and vertical lines
            if img_array.shape[0] > 10 and img_array.shape[1] > 10:
                # Horizontal edge detection
                h_edges = np.abs(img_array[1:, :] - img_array[:-1, :])
                h_line_count = np.sum(h_edges > 30) / h_edges.size
                features['horizontal_line_density'] = float(h_line_count)
                
                # Vertical edge detection  
                v_edges = np.abs(img_array[:, 1:] - img_array[:, :-1])
                v_line_count = np.sum(v_edges > 30) / v_edges.size
                features['vertical_line_density'] = float(v_line_count)
            
            return features
            
        except Exception as e:
            logger.error(f"Layout analysis error: {e}")
            return {}
            
    def process_with_ocr(self, file_path: str) -> Dict[str, Any]:
        """Process file with OCR using intelligent handler selection"""
        
        try:
            if self.log_details:
                logger.info(f"\n🔍 Processing file: {os.path.basename(file_path)}")
                logger.info(f"👤 User: {self.current_user}")
                logger.info(f"⏰ Time: {self.current_time}")
                logger.info("=" * 60)
            
            # Update current time for accurate timestamps
            self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Load image (handle PDF conversion elsewhere if needed)
            if self.get_file_type(file_path) == 'image':
                image = Image.open(file_path)
                if self.log_details:
                    logger.info(f"📐 Image loaded: {image.size[0]}x{image.size[1]} pixels")
            else:
                return {'success': False, 'error': 'PDF processing not implemented in this handler'}
            
            # DETECT LIST DOCUMENTS DIRECTLY - High priority fix for lists 
            list_confidence = self._detect_list_document(image)
            if list_confidence >= 0.7 and 'list' in self.handlers:
                # Direct list detection override
                selected_handler_type = 'list'
                selected_handler = self.handlers['list']
                confidence = list_confidence * 100
                evidence = [f"List features detected: high confidence bullet points"]
            else:
                # REGULAR DOCUMENT DETECTION
                selected_handler_type, confidence, evidence = self._detect_document_type(image)
                selected_handler = self.handlers.get(selected_handler_type)
                
                # Fall back to book handler if selected isn't available
                if not selected_handler and 'book' in self.handlers:
                    selected_handler_type = 'book'
                    selected_handler = self.handlers['book']
                    confidence = 30.0
                elif not selected_handler and self.handlers:
                    # Ultimate fallback to any available handler
                    selected_handler_type = next(iter(self.handlers))
                    selected_handler = self.handlers[selected_handler_type]
                    confidence = 20.0
            
            # Get handler name for logging
            handler_name = selected_handler.__class__.__name__
            
            # Log detection results
            if self.log_details:
                logger.info(f"📄 Document type detected: {selected_handler_type}")
                logger.info(f"🤖 Selected handler: {handler_name}")
                logger.info(f"📊 Detection confidence: {confidence:.1f}%")
                
                # Show evidence for decision
                if evidence:
                    logger.info("🔍 Detection evidence:")
                    for ev in evidence[:3]:  # Top 3 evidence items
                        logger.info(f"  • {ev}")
            
            # Process with selected handler
            logger.info(f"⚙️ Processing with {handler_name}...")
            result = selected_handler.extract_text(image, preprocess=True)
            
            # Add metadata
            result.update({
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'document_type': selected_handler_type,
                'handler_used': handler_name,
                'confidence': confidence,
                'evidence': evidence,
                'user': self.current_user,
                'timestamp': self.current_time
            })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Processing error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'user': self.current_user,
                'timestamp': self.current_time
            }
    
    def _detect_document_type(self, image: Image.Image) -> Tuple[str, float, List[str]]:
        """Detect document type based on content and visual features"""
        evidence = []
        
        # Extract text sample for content analysis
        if TESSERACT_AVAILABLE:
            try:
                small_img = image.copy()
                small_img.thumbnail((800, 600), Image.LANCZOS)
                text = pytesseract.image_to_string(small_img)
            except Exception:
                text = ""
        else:
            text = ""
        
        # Extract visual features
        visual_features = self._extract_visual_features(image)
        
        # Document type confidence scores - INCLUDES ALL HANDLERS
        type_confidence = {
            'list': self._score_list_document(text, visual_features),
            'receipt': self._score_receipt_document(text, visual_features),
            'form': self._score_form_document(text, visual_features),
            'table': self._score_table_document(text, visual_features),
            'code': self._score_code_document(text, visual_features),
            'academic': self._score_academic_document(text, visual_features),
            'pdf': self._score_pdf_document(text, visual_features),
            'title': self._score_title_document(text, visual_features),
            'book': self._score_book_document(text, visual_features),
        }
        
        # Get evidence for top document types
        for doc_type, (confidence, doc_evidence) in type_confidence.items():
            if doc_evidence:
                evidence.extend(doc_evidence[:2])  # Top 2 evidence items
        
        # Find best document type
        best_type = 'book'  # Default fallback
        best_confidence = 0.0
        
        for doc_type, (confidence, _) in type_confidence.items():
            # Only consider handlers we actually have
            if doc_type in self.handlers and confidence > best_confidence:
                best_type = doc_type
                best_confidence = confidence
        
        # If no good match, ensure we return with book handler
        if best_confidence < 30 and 'book' in self.handlers:
            return 'book', max(30.0, best_confidence), evidence
            
        return best_type, best_confidence, evidence
    
    def _score_list_document(self, text: str, visual_features: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Calculate confidence that a document is a list"""
        confidence = 0.0
        evidence = []
        
        # Check for bullet points and list markers
        bullet_count = text.count('•')
        star_count = text.count('*')
        dash_list_count = len(re.findall(r'^\s*-\s+', text, re.MULTILINE))
        number_list_count = len(re.findall(r'^\s*\d+\.\s+', text, re.MULTILINE))
        letter_list_count = len(re.findall(r'^\s*[a-z]\)\s+', text, re.MULTILINE))
        
        # Add confidence based on markers
        if bullet_count > 0:
            confidence += min(10 * bullet_count, 40)
            evidence.append(f"Found {bullet_count} bullet points (•)")
            
        if star_count > 0:
            confidence += min(8 * star_count, 30)
            evidence.append(f"Found {star_count} star bullets (*)")
            
        if dash_list_count > 0:
            confidence += min(9 * dash_list_count, 35)
            evidence.append(f"Found {dash_list_count} dash list items")
            
        if number_list_count > 0:
            confidence += min(10 * number_list_count, 40)
            evidence.append(f"Found {number_list_count} numbered list items")
            
        if letter_list_count > 0:
            confidence += min(9 * letter_list_count, 35)
            evidence.append(f"Found {letter_list_count} lettered list items")
            
        # Visual features for lists
        if visual_features.get('is_portrait', False):
            confidence += 5
        
        # Cap confidence at 95%
        return min(confidence, 95), evidence
    
    def _score_receipt_document(self, text: str, visual_features: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Calculate confidence that a document is a receipt"""
        confidence = 0.0
        evidence = []
        
        # Check for receipt-specific terms
        receipt_terms = {
            'total': 15, 'subtotal': 15, 'tax': 10, 
            'payment': 10, 'cash': 5, 'change': 5, 
            'amount': 5, 'price': 5, 'receipt': 10
        }
        
        # Add confidence for each term found
        text_lower = text.lower()
        for term, value in receipt_terms.items():
            if term in text_lower:
                confidence += value
                evidence.append(f"Found receipt term: '{term}'")
        
        # Check for currency patterns
        price_patterns = len(re.findall(r'[$£€]\s*\d+\.\d{2}', text))
        if price_patterns > 0:
            confidence += min(5 * price_patterns, 25)
            evidence.append(f"Found {price_patterns} price patterns")
            
        # Visual features - receipts are often narrow/tall
        if visual_features.get('is_narrow', False):
            confidence += 15
            evidence.append("Document has receipt-like proportions")
        
        # Cap confidence at 95%
        return min(confidence, 95), evidence
    
    def _score_form_document(self, text: str, visual_features: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Calculate confidence that a document is a form"""
        confidence = 0.0
        evidence = []
        
        # Form field patterns
        form_patterns = {
            r'name:.*?_____': 15,
            r'address:': 10, 
            r'date:.*?[_/]': 10,
            r'signature:': 15,
            r'^\s*\[\s*\]\s*': 12,  # Checkbox
            r'please\s+fill': 8
        }
        
        # Check for form patterns
        for pattern, value in form_patterns.items():
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            if matches:
                confidence += min(value * len(matches), value * 2)
                evidence.append(f"Found {len(matches)} form {pattern.strip('^$')} fields")
        
        # Visual features - forms often have horizontal lines
        if visual_features.get('horizontal_line_density', 0) > 0.03:
            confidence += 15
            evidence.append("Document has horizontal lines typical of forms")
            
        # Cap confidence at 95%
        return min(confidence, 95), evidence
    
    def _score_table_document(self, text: str, visual_features: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Calculate confidence that a document is a table"""
        confidence = 0.0
        evidence = []
        
        # Table patterns
        table_patterns = {
            r'\|\s*\w+\s*\|': 15,  # Cell with content
            r'\+----': 12,         # Table border
            r'-{3,}\+': 12,        # Table separator
            r'^\s*\|.*\|\s*$': 15  # Row
        }
        
        # Check for table patterns
        for pattern, value in table_patterns.items():
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                confidence += min(value * len(matches), value * 2)
                evidence.append(f"Found {len(matches)} table structure elements")
        
        # Visual features - tables have grid structure
        if visual_features.get('horizontal_line_density', 0) > 0.03 and \
           visual_features.get('vertical_line_density', 0) > 0.03:
            confidence += 30
            evidence.append("Document has grid structure typical of tables")
            
        # Cap confidence at 95%
        return min(confidence, 95), evidence
    
    def _score_code_document(self, text: str, visual_features: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Calculate confidence that a document is code"""
        confidence = 0.0
        evidence = []
        
        # Code patterns
        code_patterns = {
            r'def\s+\w+\(': 15,      # Python function
            r'class\s+\w+[:(]': 15,   # Python/Java/C# class
            r'function\s+\w+\(': 15,  # JavaScript function
            r'import\s+\w+': 10,      # Python/Java import
            r'^\s*\/\/': 8,           # Line comment
            r'\/\*.*?\*\/': 8,        # Block comment
            r'{\s*$': 7,              # Opening brace
            r'}\s*$': 7               # Closing brace
        }
        
        # Check for code patterns
        for pattern, value in code_patterns.items():
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                confidence += min(value * len(matches), value * 2)
                evidence.append(f"Found {len(matches)} code {pattern.strip('^$')} elements")
        
        # Cap confidence at 95%
        return min(confidence, 95), evidence
    
    def _score_academic_document(self, text: str, visual_features: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Calculate confidence that a document is academic/scholarly"""
        confidence = 0.0
        evidence = []
        
        # Academic document patterns
        academic_patterns = {
            r'abstract': 12,
            r'introduction': 8, 
            r'conclusion': 8,
            r'references': 10,
            r'et\s+al\.': 15,     # Citations
            r'\(\d{4}\)': 10,      # Year citations
            r'fig\.\s*\d+': 10,    # Figure references
            r'table\s*\d+': 10     # Table references
        }
        
        # Check for academic patterns
        for pattern, value in academic_patterns.items():
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            if matches:
                confidence += min(value * len(matches), value * 2)
                evidence.append(f"Found {len(matches)} academic {pattern.strip('^$')} elements")
        
        # Cap confidence at 95%
        return min(confidence, 95), evidence
        
    def _score_pdf_document(self, text: str, visual_features: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Calculate confidence that a document is a PDF"""
        confidence = 0.0
        evidence = []
        
        # Check for standard page format
        if visual_features.get('is_standard_paper', False):
            confidence += 10
            evidence.append("Document has standard PDF-like proportions")
            
        # PDF often has page numbers
        page_numbers = len(re.findall(r'page\s*\d+|p\.\s*\d+', text, re.IGNORECASE))
        if page_numbers > 0:
            confidence += min(5 * page_numbers, 15)
            evidence.append(f"Found {page_numbers} page number references")
            
        # Look for PDF metadata markers
        pdf_markers = ['pdf', 'adobe', 'acrobat', 'document']
        for marker in pdf_markers:
            if marker.lower() in text.lower():
                confidence += 10
                evidence.append(f"Found PDF marker: '{marker}'")
                break
        
        # Start with base confidence for PDFs
        if confidence == 0:
            confidence = 10  # Low default confidence
            evidence.append("Default PDF confidence level")
            
        # Cap confidence at 95%
        return min(confidence, 95), evidence
        
    def _score_title_document(self, text: str, visual_features: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Calculate confidence that a document is a title page"""
        confidence = 0.0
        evidence = []
        
        # Title pages often have large text and centered content
        # Check for title-like features
        
        # Check for very short document (title pages are often brief)
        if len(text.split()) < 50:
            confidence += 20
            evidence.append("Document is brief, typical of title pages")
            
        # Check for title-like patterns
        title_patterns = {
            r'title:': 15,
            r'by:?\s*\w+': 10,  # Author attribution
            r'copyright': 10,
            r'all\s+rights\s+reserved': 10,
            r'presented\s+by|prepared\s+by': 15,
            r'university|college': 10,
            r'thesis|dissertation': 15,
            r'volume': 10
        }
        
        # Check for title patterns
        for pattern, value in title_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                confidence += value
                evidence.append(f"Found title page pattern: '{pattern}'")
        
        # Title pages often have distinct visual characteristics
        if visual_features.get('text_density', 0) < 0.1:
            confidence += 15
            evidence.append("Document has sparse text typical of title pages")
            
        # Cap confidence at 95%
        return min(confidence, 95), evidence
    
    def _score_book_document(self, text: str, visual_features: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Calculate confidence that a document is a book page"""
        confidence = 0.0
        evidence = []
        
        # Book page patterns
        book_patterns = {
            r'chapter\s*\d+': 12,
            r'page\s*\d+': 8,
            r'section\s*\d+\.\d+': 10,
            r'contents': 9,
            r'index': 9,
            r'glossary': 8
        }
        
        # Check for book patterns
        for pattern, value in book_patterns.items():
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            if matches:
                confidence += value * len(matches)
                evidence.append(f"Found {len(matches)} book {pattern.strip('^$')} elements")
        
        # Visual features - book pages often have standard dimensions
        if visual_features.get('is_standard_paper', False):
            confidence += 10
            evidence.append("Document has standard page proportions")
            
        # Dense text is typical of books
        if visual_features.get('text_density', 0) > 0.15:
            confidence += 10
            evidence.append("Document has dense text typical of books")
            
        # Start with base confidence for book (default type)
        if confidence == 0:
            confidence = 30
            evidence.append("Default document type")
            
        # Cap confidence at 95%
        return min(confidence, 95), evidence

    def set_debug_mode(self, debug_mode: bool):
        """Set debug mode for all components"""
        self.debug_mode = debug_mode
        
        # Set debug on all handlers
        for handler in self.handlers.values():
            if hasattr(handler, 'set_debug_mode'):
                handler.set_debug_mode(debug_mode)