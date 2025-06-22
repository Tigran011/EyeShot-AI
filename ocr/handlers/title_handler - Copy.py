# src/ocr/handlers/title_handler.py
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import pytesseract
import re
import time
from PIL import Image

from .base_handler import DocumentHandler

class TitleHandler(DocumentHandler):
    """Handler for stylized titles and headings with special formatting preservation"""
    
    def can_handle(self, image: Image.Image) -> bool:
        """Check if image contains a stylized title or heading"""
        try:
            # Get image dimensions
            width, height = image.size
            
            # Title images typically have specific characteristics:
            # 1. Wide aspect ratio (typically wider than tall)
            # 2. Large text compared to image size
            # 3. Centered text
            # 4. Often with decorative elements or special formatting
            
            # Check aspect ratio - titles are typically wider than tall
            aspect_ratio = width / height
            if aspect_ratio < 1.2:  # Not wide enough for a typical title
                return False
            
            # Convert to numpy array for analysis
            img_array = np.array(image.convert('L'))
            
            # Apply simple binarization to isolate text
            _, binary = cv2.threshold(img_array, 180, 255, cv2.THRESH_BINARY_INV)
            
            # Check for text content using OCR
            if 'tesseract_available' in self.engines and self.engines['tesseract_available']:
                # Create a small version for quick analysis
                small_img = image.copy()
                small_img.thumbnail((800, 200), Image.Resampling.LANCZOS)
                
                # Use PSM 7 (single line) which works well for titles
                text = pytesseract.image_to_string(small_img, config='--psm 7').strip()
                
                # Title typically has few words (less than 10-12 words)
                word_count = len(text.split())
                if 0 < word_count <= 12:
                    # Good candidate for a title
                    return True
                    
                # If longer text, check for multi-line title using PSM 6
                if word_count > 12 or word_count == 0:
                    text = pytesseract.image_to_string(small_img, config='--psm 6').strip()
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    
                    # Title images typically have 1-3 lines of text
                    if 1 <= len(lines) <= 3:
                        total_words = sum(len(line.split()) for line in lines)
                        if total_words <= 15:  # Title with multiple lines
                            return True
            
            # Visual analysis for title characteristics
            
            # 1. Check text distribution (titles typically have concentrated text in the center)
            h_projection = np.sum(binary, axis=0)  # Horizontal projection
            v_projection = np.sum(binary, axis=1)  # Vertical projection
            
            # Normalize projections
            if np.max(h_projection) > 0:
                h_projection = h_projection / np.max(h_projection)
            if np.max(v_projection) > 0:
                v_projection = v_projection / np.max(v_projection)
            
            # Titles typically have a concentration of text in the center
            # Calculate center of mass for horizontal projection
            h_center = np.sum(h_projection * np.arange(len(h_projection))) / np.sum(h_projection) if np.sum(h_projection) > 0 else width/2
            h_center_ratio = h_center / width
            
            # Center of mass should be close to the center for titles
            if 0.4 <= h_center_ratio <= 0.6:
                # Check density of text (titles typically have larger text)
                text_area = np.sum(binary > 0)
                text_ratio = text_area / (width * height)
                
                # Titles typically have a moderate amount of text covering the image
                if 0.05 <= text_ratio <= 0.4:
                    return True
            
            return False
            
        except Exception as e:
            if self.debug_mode:
                print(f"Title detection error: {e}")
            return False
    
    def _perform_extraction(self, image: Image.Image, preprocess: bool) -> Dict:
        """Extract text from stylized title with format preservation"""
        try:
            # Use specialized title preprocessing if requested
            if preprocess and 'image' in self.processors:
                processed_image = self.processors['image'].preprocess_stylized_title(image.copy())
            else:
                processed_image = image.copy()
            
            # Save debug image if enabled
            if hasattr(self, 'debug_mode') and self.debug_mode and hasattr(self, 'save_debug_images') and self.save_debug_images:
                debug_path = os.path.join(self.debug_dir, f"title_processed_{int(time.time())}.png")
                processed_image.save(debug_path)
            
            results = []
            
            # Try multiple approaches and select the best result
            
            # 1. Tesseract with specific PSM modes for titles
            if 'tesseract_available' in self.engines and self.engines['tesseract_available']:
                # Try multiple PSM values optimized for titles
                for psm in [7, 13, 6, 11]:  # In order of preference for titles
                    config = f'--oem 3 --psm {psm}'
                    
                    try:
                        text = pytesseract.image_to_string(processed_image, config=config).strip()
                        
                        if text:
                            # Calculate confidence
                            data = pytesseract.image_to_data(
                                processed_image, 
                                config=config,
                                output_type=pytesseract.Output.DICT
                            )
                            
                            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                            
                            # For titles, prefer results with more words (but not too many)
                            word_count = len(text.split())
                            word_score = min(word_count * 2, 20)  # Up to 20 points for word count
                            
                            # Calculate layout score (how well the text fits the title pattern)
                            layout_score = self._calculate_title_layout_score(text)
                            
                            # Combine scores
                            total_score = avg_confidence * 0.6 + word_score + layout_score
                            
                            results.append({
                                'text': text,
                                'confidence': avg_confidence,
                                'word_count': word_count,
                                'char_count': len(text),
                                'success': True,
                                'engine': f'tesseract_psm_{psm}',
                                'score': total_score
                            })
                    except Exception as e:
                        if self.debug_mode:
                            print(f"Title extraction with Tesseract PSM {psm} failed: {e}")
            
            # 2. Try with EasyOCR if available
            if ('easyocr_available' in self.engines and 
                self.engines.get('easyocr_available') and 
                'easyocr_reader' in self.engines):
                try:
                    # Convert to CV2 format
                    cv_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
                    
                    # Run EasyOCR with specific settings for titles
                    easyocr_results = self.engines['easyocr_reader'].readtext(
                        cv_image,
                        paragraph=False,  # Titles are typically single lines
                        detail=1,         # Get detailed results
                        width_ths=1.0,    # More tolerant width threshold
                        height_ths=1.0    # More tolerant height threshold
                    )
                    
                    if easyocr_results:
                        # Process results
                        lines = []
                        total_confidence = 0
                        
                        # Group text by vertical position
                        y_clusters = self._cluster_positions([box[0][1] for _, box, _ in easyocr_results], threshold=20)
                        
                        # Create a line for each cluster
                        for cluster in y_clusters:
                            # Get all texts in this cluster
                            cluster_texts = []
                            for (box, text, conf) in easyocr_results:
                                if box[0][1] in cluster:
                                    cluster_texts.append((box[0][0], text, conf))
                            
                            # Sort by x-position (left to right)
                            cluster_texts.sort(key=lambda x: x[0])
                            
                            # Join texts and add to lines
                            line_text = ' '.join(text for _, text, _ in cluster_texts)
                            lines.append(line_text)
                            
                            # Track confidence
                            avg_conf = sum(conf for _, _, conf in cluster_texts) / len(cluster_texts)
                            total_confidence += avg_conf
                        
                        # Join lines
                        title_text = '\n'.join(lines)
                        
                        # Calculate average confidence (0-100 scale)
                        avg_confidence = (total_confidence / len(y_clusters)) * 100 if y_clusters else 0
                        
                        # Calculate score
                        word_count = len(title_text.split())
                        word_score = min(word_count * 2, 20)
                        layout_score = self._calculate_title_layout_score(title_text)
                        total_score = avg_confidence * 0.6 + word_score + layout_score
                        
                        results.append({
                            'text': title_text,
                            'confidence': avg_confidence,
                            'word_count': word_count,
                            'char_count': len(title_text),
                            'success': True,
                            'engine': 'easyocr_title',
                            'score': total_score
                        })
                except Exception as e:
                    if self.debug_mode:
                        print(f"Title extraction with EasyOCR failed: {e}")
            
            # 3. Try with PaddleOCR if available
            if ('paddle_available' in self.engines and 
                self.engines.get('paddle_available') and 
                'paddle_ocr' in self.engines):
                try:
                    # Convert to CV2 format
                    cv_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
                    
                    # Run PaddleOCR
                    paddle_results = self.engines['paddle_ocr'].ocr(cv_image, cls=True)
                    
                    if paddle_results and len(paddle_results) > 0 and paddle_results[0]:
                        title_parts = []
                        total_confidence = 0
                        count = 0
                        
                        for line in paddle_results[0]:
                            text, confidence = line[1]
                            title_parts.append(text)
                            total_confidence += confidence
                            count += 1
                        
                        title_text = " ".join(title_parts)
                        avg_confidence = (total_confidence / count) * 100 if count > 0 else 0
                        
                        # Calculate score
                        word_count = len(title_text.split())
                        word_score = min(word_count * 2, 20)
                        layout_score = self._calculate_title_layout_score(title_text)
                        total_score = avg_confidence * 0.6 + word_score + layout_score
                        
                        results.append({
                            'text': title_text,
                            'confidence': avg_confidence,
                            'word_count': word_count,
                            'char_count': len(title_text),
                            'success': True,
                            'engine': 'paddleocr_title',
                            'score': total_score
                        })
                except Exception as e:
                    if self.debug_mode:
                        print(f"Title extraction with PaddleOCR failed: {e}")
            
            # Choose best result
            if results:
                best_result = max(results, key=lambda x: x.get('score', 0))
                
                # Apply title-specific text cleanup
                if 'text' in self.processors:
                    best_result['text'] = self.processors['text'].clean_title_text(best_result['text'])
                else:
                    # Apply basic title formatting if processor not available
                    best_result['text'] = self._clean_title_text(best_result['text'])
                
                # Apply proper title case
                best_result['text'] = self._apply_title_case(best_result['text'])
                
                # Add title-specific metadata
                best_result['is_title'] = True
                best_result['has_line_breaks'] = '\n' in best_result['text']
                
                return best_result
            else:
                return self._error_result("Title text extraction failed - no valid results")
                
        except Exception as e:
            return self._error_result(f"Title extraction error: {str(e)}")

    def _clean_title_text(self, text: str) -> str:
        """Clean and normalize title text"""
        if not text:
            return ""
        
        # For titles, be very conservative with changes
        
        # Remove extra line breaks
        text = re.sub(r'\n+', '\n', text.strip())
        
        # Remove excess whitespace but preserve line breaks
        lines = []
        for line in text.split('\n'):
            lines.append(' '.join(part for part in line.split() if part))
        
        # Join lines with proper line breaks
        text = '\n'.join(lines)
        
        # Fix common title OCR errors
        replacements = {
            # Common OCR errors in titles
            '|': 'I',         # Vertical bar to I
            'l.': 'i.',       # lowercase L with period to i
            'rn': 'm',        # 'rn' is often misrecognized as 'm'
            'cl': 'd',        # 'cl' is often misrecognized as 'd'
            '0': 'O',         # In titles, 0 is often actually an O
            'Ouidon': 'Guidon',  # Common error
            'Guldel': 'Guide',   # Common error
            'tvlanual': 'Manual', # Common error
            ' :': ':',        # Fix spacing around punctuation
            ' .': '.',
            ' ,': ',',
            ' !': '!'
        }
        
        # Apply replacements
        for error, correction in replacements.items():
            text = text.replace(error, correction)
        
        return text

    def _apply_title_case(self, text: str) -> str:
        """Apply proper title case formatting to title text"""
        if not text:
            return ""
            
        # Process line by line (for multi-line titles)
        result_lines = []
        for line in text.split('\n'):
            # Split into words
            words = line.split()
            if not words:
                result_lines.append('')
                continue
                
            # Always capitalize first and last word
            if words[0]:
                words[0] = words[0][0].upper() + words[0][1:] if len(words[0]) > 1 else words[0].upper()
            if len(words) > 1 and words[-1]:
                words[-1] = words[-1][0].upper() + words[-1][1:] if len(words[-1]) > 1 else words[-1].upper()
            
            # Articles, conjunctions, and prepositions to keep lowercase unless they're first/last
            lowercase_words = {'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'on', 'at', 
                              'to', 'by', 'from', 'in', 'of', 'with'}
            
            # Process remaining words
            for i in range(1, len(words) - 1):
                word = words[i]
                if not word:
                    continue
                    
                word_lower = word.lower()
                
                # Skip words that are already in ALL CAPS (likely acronyms)
                if word.isupper():
                    continue
                
                # Keep lowercase words lowercase
                if word_lower in lowercase_words:
                    words[i] = word_lower
                else:
                    # Capitalize first letter of other words
                    words[i] = word[0].upper() + word[1:] if len(word) > 1 else word.upper()
            
            # Join words back into line
            result_lines.append(' '.join(words))
        
        # Join lines
        return '\n'.join(result_lines)

    def _calculate_title_layout_score(self, text: str) -> float:
        """Calculate a score based on how well the text fits title layout patterns"""
        if not text:
            return 0
        
        score = 0
        
        # Title characteristics and their scores
        
        # 1. Proper length (not too long, not too short)
        word_count = len(text.split())
        if 2 <= word_count <= 10:
            score += 10  # Optimal title length
        elif 1 <= word_count <= 15:
            score += 5   # Still reasonable title length
        
        # 2. Has some capitalized words (titles typically have capitalization)
        capital_word_count = sum(1 for word in text.split() if word and word[0].isupper())
        if capital_word_count >= min(2, max(1, word_count // 2)):
            score += 10  # Good capitalization pattern for a title
        
        # 3. No lowercase paragraph beginnings (titles don't start paragraphs with lowercase)
        lines = text.split('\n')
        if all(not line or line[0].isupper() for line in lines):
            score += 5
        
        # 4. Limited punctuation (titles typically have minimal punctuation)
        punctuation_count = sum(1 for c in text if c in ',.;:!?()[]{}')
        if punctuation_count <= 2:
            score += 5
        elif punctuation_count > 5:
            score -= 5  # Too much punctuation reduces title probability
        
        # 5. No trailing periods unless it's part of abbreviation (titles don't end with periods)
        if not text.strip().endswith('.') or re.search(r'\b[A-Z]\.$', text):
            score += 5
        
        return score

    def _cluster_positions(self, positions: List[float], threshold: float = 10) -> List[List[float]]:
        """Group positions that are close to each other"""
        if not positions:
            return []
        
        # Sort positions
        positions = sorted(positions)
        
        # Initialize clusters with first position
        clusters = [[positions[0]]]
        
        # Cluster positions
        for pos in positions[1:]:
            # Check if position belongs to the last cluster
            if pos - clusters[-1][-1] < threshold:
                # Add to existing cluster
                clusters[-1].append(pos)
            else:
                # Start new cluster
                clusters.append([pos])
        
        return clusters

    def _extract_title_with_line_breaks(self, image: Image.Image, preprocess: bool) -> Dict:
        """Extract title text while preserving line breaks"""
        # First use the regular title extraction
        result = self._perform_extraction(image, preprocess)
        
        if not result['success']:
            return result
        
        # Now detect the line breaks
        try:
            # Convert to OpenCV format
            if preprocess and 'image' in self.processors:
                processed = self.processors['image'].preprocess_stylized_title(image.copy())
            else:
                processed = image.copy()
                
            img_array = np.array(processed.convert('L'))
            
            # Find text blocks
            if 'tesseract_available' in self.engines and self.engines['tesseract_available']:
                # Get detailed data with bounding boxes
                data = pytesseract.image_to_data(
                    processed, 
                    config='--oem 3 --psm 6',
                    output_type=pytesseract.Output.DICT
                )
                
                # Group words by their y-coordinate (lines)
                lines = {}
                for i, text in enumerate(data['text']):
                    if not text.strip():
                        continue
                        
                    top = data['top'][i]
                    # Group within 15 pixels vertically
                    line_key = top // 15
                    if line_key not in lines:
                        lines[line_key] = []
                    lines[line_key].append((data['left'][i], text))
                
                # Sort by y-coordinate and join lines
                sorted_lines = []
                for line_key in sorted(lines.keys()):
                    # Sort words in line by x-coordinate
                    sorted_words = sorted(lines[line_key], key=lambda x: x[0])
                    sorted_lines.append(" ".join(word for _, word in sorted_words))
                
                # Create the properly structured title
                if sorted_lines:
                    structured_title = "\n".join(sorted_lines)
                    result['text'] = structured_title
                    result['has_line_breaks'] = True
            
            return result
            
        except Exception as e:
            # If structure detection fails, return the original result
            if self.debug_mode:
                print(f"Title structure detection error: {e}")
            return result

    def _error_result(self, message: str) -> Dict:
        """Create a standardized error result"""
        return {
            'text': '',
            'confidence': 0,
            'word_count': 0,
            'char_count': 0,
            'success': False,
            'error': message,
            'engine': 'title_extraction_failed'
        }