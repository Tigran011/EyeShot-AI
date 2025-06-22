"""
EyeShot AI - Enhanced Receipt Handler
Specialized in extracting structured text from receipts and invoices
Author: Tigran0000
Last updated: 2025-06-22 13:57:52 UTC
"""

import os
import re
import time
import numpy as np
import cv2
import pytesseract
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from .base_handler import DocumentHandler

class ReceiptHandler(DocumentHandler):
    """Handler for receipts and invoices with enhanced structure preservation"""
    
    def __init__(self):
        super().__init__()
        self.document_type = "receipt"
        self.name = "receipt"
        self.description = "Handler for receipts and invoices"
        
        # Receipt-specific detection parameters
        self.line_tolerance = 8  # Pixels tolerance for same line - tighter for receipts
        self.section_gap_threshold = 25  # Pixels for section separation in receipts
        self.item_indent_threshold = 15  # Pixels for item indentation detection
        
        # Structure detection parameters
        self.column_detection_threshold = 3  # Minimum occurrences for column detection
        self.price_column_width = 12  # Standard width for price columns
        
        # Symbol mappings for receipt-specific elements
        self.receipt_symbol_fixes = {
            'S': '$',             # When preceding a digit
            '€o.': '€0.',         # Euro symbol ocr errors
            '£o.': '£0.',         # Pound symbol ocr errors
            '$o.': '$0.',         # Dollar symbol ocr errors
            'O.': '0.',           # Letter O to number 0
            'o.': '0.',           # Letter o to number 0
            'I.': '1.',           # Letter I to number 1
            'l.': '1.',           # Letter l to number 1
        }
        
        # Receipt item bullet mappings
        self.item_bullet_fixes = {
            'e ': '• ',
            '« ': '• ',
            '* ': '• ',
            '- ': '• ',
            '> ': '• ',
            '» ': '• ',
        }
    
    def can_handle(self, image: Image.Image) -> Dict[str, Any]:
        """Check if image is likely a receipt based on visual characteristics"""
        try:
            # Check for typical receipt aspect ratio
            width, height = image.size
            aspect_ratio = width / height
            
            # Convert to grayscale for analysis
            img_array = np.array(image.convert('L'))
            
            # Check brightness - receipts typically have very white backgrounds
            avg_brightness = np.mean(img_array)
            
            # Receipt indicator check with OCR
            indicators = self._check_receipt_indicators(image)
            indicator_score = indicators.get('score', 0)
            
            # Structure check
            structure_score = self._check_receipt_structure(img_array, width, height)
            
            # Calculate confidence based on multiple factors
            confidence = 0
            
            # Aspect ratio scoring
            if aspect_ratio > 1.4:  # Wide format (horizontal receipt)
                confidence += 20
            elif aspect_ratio < 0.7:  # Narrow format (traditional receipt)
                confidence += 30
            else:  # Standard document
                confidence += 10
                
            # Brightness scoring
            if avg_brightness > 200:  # Very white (typical for receipts)
                confidence += 15
            elif avg_brightness > 180:  # Fairly white
                confidence += 10
            
            # Indicator scoring (0-40 points)
            confidence += min(indicator_score * 8, 40)
            
            # Structure scoring (0-30 points)
            confidence += min(structure_score * 10, 30)
            
            # Cap at 95%
            confidence = min(confidence, 95)
            
            return {
                'can_handle': confidence > 60,
                'confidence': confidence,
                'indicators': indicators.get('found', []),
                'aspect_ratio': aspect_ratio,
                'brightness': avg_brightness,
                'structure_score': structure_score
            }
            
        except Exception as e:
            return {
                'can_handle': False,
                'confidence': 0,
                'error': str(e)
            }
    
    def _check_receipt_indicators(self, image: Image.Image) -> Dict[str, Any]:
        """Check for receipt content indicators using lightweight OCR"""
        try:
            # Create a small version for quick analysis
            small_img = image.copy()
            small_img.thumbnail((600, 800), Image.LANCZOS)
            
            # Quick OCR with simple config
            text = pytesseract.image_to_string(small_img, config='--psm 6').lower()
            
            # Receipt indicators with assigned weights
            receipt_indicators = {
                'total': 2,       # Strong indicators
                'subtotal': 2,
                'tax': 1.5,
                'amount': 1.5,
                'change': 1.5,
                'cash': 1,
                'receipt': 2,
                'order': 1,
                'item': 1,
                'qty': 1.5,
                'price': 1.5,
                'payment': 1.5,
                'store': 1,
                'merchant': 1.5,
                'cashier': 1.5,
                'customer': 1,
                'sale': 1,
                'invoice': 1.5,
                'thank you': 1,
                'transaction': 1.5,
            }
            
            # Check for currency symbols (strong indicators)
            currency_pattern = r'[$£€¥]\s*\d+[\.,]\d{2}'
            currency_matches = re.findall(currency_pattern, text)
            
            found_indicators = []
            indicator_score = 0
            
            # Check for text indicators
            for indicator, weight in receipt_indicators.items():
                if indicator in text:
                    found_indicators.append(indicator)
                    indicator_score += weight
            
            # Add currency symbol score
            if currency_matches:
                found_indicators.append('currency_symbols')
                indicator_score += min(len(currency_matches), 5) * 1.5
            
            return {
                'score': indicator_score,
                'found': found_indicators,
                'currency_matches': len(currency_matches)
            }
            
        except Exception:
            return {'score': 0, 'found': []}
    
    def _check_receipt_structure(self, img_array, width, height) -> float:
        """Check for receipt-like structure"""
        try:
            # Look for horizontal lines and aligned blocks of text
            structure_score = 0
            
            # 1. Edge detection for finding lines
            edges = cv2.Canny(img_array, 50, 200)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                   minLineLength=width*0.3, maxLineGap=10)
            
            horizontal_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Check if line is horizontal
                    if abs(y2 - y1) < 10:
                        horizontal_lines += 1
            
            # 2. Text layout analysis using horizontal projection
            h_projection = np.sum(edges, axis=1)
            text_lines = [i for i, val in enumerate(h_projection) if val > width*0.1]
            
            # Group adjacent lines to find text blocks
            text_blocks = []
            current_block = []
            
            for i in range(len(text_lines)):
                if not current_block or text_lines[i] - current_block[-1] <= 15:
                    current_block.append(text_lines[i])
                else:
                    text_blocks.append(current_block)
                    current_block = [text_lines[i]]
            
            if current_block:
                text_blocks.append(current_block)
            
            # 3. Analyze vertical spacing consistency
            if len(text_blocks) >= 3:
                # Calculate gaps between blocks
                gaps = [text_blocks[i+1][0] - text_blocks[i][-1] 
                       for i in range(len(text_blocks)-1)]
                
                # Check for consistent gaps (receipt items often have uniform spacing)
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    gap_consistency = sum(1 for gap in gaps 
                                        if abs(gap - avg_gap) < 5) / len(gaps)
                    
                    # Highly consistent spacing is typical for receipts
                    structure_score += gap_consistency * 1.5
            
            # Score based on horizontal lines (separators in receipts)
            structure_score += min(horizontal_lines / 3, 1)
            
            # Score based on text block count (receipts have many lines)
            structure_score += min(len(text_blocks) / 8, 1)
            
            return structure_score
            
        except Exception:
            return 0
    
    def extract_text(self, image: Image.Image, preprocess: bool = True) -> Dict[str, Any]:
        """Extract text from receipt with structure preservation"""
        try:
            # Measure processing time
            start_time = time.time()
            
            # Use specialized receipt preprocessing if requested
            if preprocess and hasattr(self, 'processors') and self.processors.get('image'):
                processed_image = self.processors['image'].preprocess_receipt(image.copy())
            else:
                processed_image = self._preprocess_receipt_image(image.copy())
            
            # Get OCR data with detailed layout info
            data = pytesseract.image_to_data(
                processed_image,
                config='--oem 3 --psm 6 -c preserve_interword_spaces=1',
                output_type=pytesseract.Output.DICT
            )
            
            # Analyze receipt structure
            structure_info = self._analyze_receipt_structure(data, processed_image.size)
            
            # Build structured text
            text = self._build_structured_receipt_text(data, structure_info)
            
            # Extract receipt metadata
            receipt_info = self._extract_receipt_info(text)
            
            # Calculate confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Processing time
            processing_time = time.time() - start_time
            
            return {
                'text': text,
                'confidence': confidence,
                'word_count': len(text.split()),
                'char_count': len(text),
                'success': True,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': processing_time,
                'receipt_info': receipt_info,
                'structure': structure_info['type'],
                'columns': structure_info['columns'],
                'user': 'Tigran0000'
            }
            
        except Exception as e:
            return {
                'text': '',
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'user': 'Tigran0000'
            }
    
    def _preprocess_receipt_image(self, image: Image.Image) -> Image.Image:
        """Apply receipt-specific preprocessing"""
        try:
            # Convert to numpy array for OpenCV operations
            img_array = np.array(image)
            
            # Convert to grayscale if it's not already
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply adaptive thresholding to handle varying lighting
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 21, 11
            )
            
            # Noise removal
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Convert back to PIL
            return Image.fromarray(opening)
            
        except Exception:
            # Return original if preprocessing fails
            return image
    
    def _analyze_receipt_structure(self, data: Dict, image_size: Tuple[int, int]) -> Dict:
        """Analyze receipt structure for better text formatting"""
        
        # Extract valid text elements with positions
        elements = []
        for i in range(len(data['text'])):
            if data['conf'][i] > 20 and data['text'][i].strip():  # Lower threshold for receipts
                elements.append({
                    'text': data['text'][i].strip(),
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i],
                    'conf': data['conf'][i],
                    'block': data['block_num'][i],
                    'paragraph': data['par_num'][i],
                    'line': data['line_num'][i]
                })
        
        # Sort elements by reading order (top to bottom, left to right)
        elements.sort(key=lambda e: (e['y'], e['x']))
        
        # Detect receipt sections
        header, body, footer = self._detect_receipt_sections(elements, image_size[1])
        
        # Detect columns
        columns = self._detect_receipt_columns(elements)
        
        # Detect if it's an itemized receipt
        is_itemized = self._detect_itemized_receipt(elements, columns)
        
        # Determine overall structure type
        if is_itemized and len(columns) >= 2:
            structure_type = "itemized_columnar"
        elif len(columns) >= 2:
            structure_type = "columnar"
        elif is_itemized:
            structure_type = "itemized_simple"
        else:
            structure_type = "simple"
        
        return {
            'type': structure_type,
            'elements': elements,
            'columns': columns,
            'is_itemized': is_itemized,
            'sections': {
                'header': header,
                'body': body,
                'footer': footer
            }
        }
    
    def _detect_receipt_sections(self, elements, image_height):
        """Detect header, body, and footer sections of a receipt"""
        if not elements:
            return [], [], []
        
        # Get unique y positions to analyze distribution
        y_positions = sorted(set(e['y'] for e in elements))
        
        # Calculate section boundaries based on content distribution
        header_end = y_positions[min(len(y_positions) // 5, 5)] if len(y_positions) > 5 else 0
        footer_start = y_positions[max(len(y_positions) - len(y_positions) // 4, 0)] if len(y_positions) > 4 else image_height
        
        # Assign elements to sections
        header = [e for e in elements if e['y'] <= header_end]
        footer = [e for e in elements if e['y'] >= footer_start]
        body = [e for e in elements if header_end < e['y'] < footer_start]
        
        return header, body, footer
    
    def _detect_receipt_columns(self, elements):
        """Detect columns in receipt based on x-coordinate clustering"""
        if not elements:
            return []
        
        # Collect x positions
        x_positions = [e['x'] for e in elements]
        
        # Create position histogram with clustering
        position_counts = {}
        for pos in x_positions:
            # Group positions within 20 pixels
            pos_key = (pos // 20) * 20
            position_counts[pos_key] = position_counts.get(pos_key, 0) + 1
        
        # Find significant column positions (occurring multiple times)
        min_count = max(3, len(elements) // 10)  # Minimum frequency to be a column
        significant_columns = [(pos, count) for pos, count in position_counts.items() 
                             if count >= min_count]
        
        # Sort columns by position (left to right)
        significant_columns.sort()
        
        # Extract just the positions
        columns = [pos for pos, _ in significant_columns]
        
        # Check if last column might be a price column (common in receipts)
        if len(columns) >= 2:
            # If last column has significantly less text, it might be prices
            last_col_elements = [e for e in elements if abs(e['x'] - columns[-1]) < 30]
            avg_width_last = sum(e['w'] for e in last_col_elements) / len(last_col_elements) if last_col_elements else 0
            
            if avg_width_last < 60:  # Narrow column, likely prices
                columns[-1] = ('price_column', columns[-1])
        
        return columns
    
    def _detect_itemized_receipt(self, elements, columns):
        """Detect if receipt has itemized entries (product list with prices)"""
        if not elements or len(elements) < 5:
            return False
        
        # Look for patterns indicating itemized entries
        # 1. Quantity-like numbers at start of line
        qty_pattern = re.compile(r'^\d+(\.\d*)?$')  # Just a number like "2" or "2.5"
        
        # 2. Price-like numbers
        price_pattern = re.compile(r'\d+\.\d{2}$')  # Format like "12.99"
        
        # Count likely item lines
        item_lines = 0
        price_elements = 0
        
        # Group elements by line
        lines = {}
        for e in elements:
            line_key = f"{e['block']}_{e['line']}"
            if line_key not in lines:
                lines[line_key] = []
            lines[line_key].append(e)
        
        # Analyze each line
        for line_elements in lines.values():
            line_elements.sort(key=lambda e: e['x'])
            
            # Check for price at end of line
            if line_elements and price_pattern.match(line_elements[-1]['text']):
                price_elements += 1
                
                # Check if first element might be a quantity
                if len(line_elements) > 1 and qty_pattern.match(line_elements[0]['text']):
                    item_lines += 1
                # Or if there's just item description and price
                elif len(line_elements) > 1:
                    item_lines += 1
        
        # If we have several likely item lines, it's probably itemized
        return item_lines >= 3
 
    def _build_receipt_structure(self, data):
        """Grid-based structure preservation for receipts"""
        
        # Create a character grid with the exact dimensions of the original
        width = max(data['left'][i] + data['width'][i] for i in range(len(data['text']))) + 10
        height = max(data['top'][i] + data['height'][i] for i in range(len(data['text']))) + 10
        
        # Initialize empty grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Place each character at its exact position
        for i in range(len(data['text'])):
            if not data['text'][i].strip():
                continue
                
            text = data['text'][i]
            x = data['left'][i]
            y = data['top'][i]
            
            # Place text at exact coordinates
            for j, char in enumerate(text):
                if y < height and x+j < width:
                    grid[y][x+j] = char
        
        # Convert grid to lines with exact positioning
        result_lines = []
        for y in range(height):
            # Get line content preserving all spaces
            line = ''.join(grid[y])
            # Remove trailing spaces only
            line = line.rstrip()
            result_lines.append(line)
        
        # Remove empty lines at start/end but preserve internal empty lines
        while result_lines and not result_lines[0].strip():
            result_lines.pop(0)
        while result_lines and not result_lines[-1].strip():
            result_lines.pop()
        
        return '\n'.join(result_lines)
    
    def _build_structured_receipt_text(self, data, structure_info):
        """Build text with structure preservation based on receipt layout"""
        elements = structure_info['elements']
        if not elements:
            return ""
        
        # Group elements into lines
        lines = self._group_elements_into_lines(elements)
        
        # Process lines according to receipt structure
        if structure_info['type'] in ['itemized_columnar', 'columnar']:
            return self._format_columnar_receipt(lines, structure_info)
        else:
            return self._format_simple_receipt(lines, structure_info)
    
    def _group_elements_into_lines(self, elements):
        """Group elements into logical text lines based on y-position"""
        if not elements:
            return []
        
        lines = []
        current_line = [elements[0]]
        current_y = elements[0]['y']
        
        for element in elements[1:]:
            # Calculate if element belongs to current line
            y_diff = abs(element['y'] - current_y)
            
            if y_diff <= self.line_tolerance:
                current_line.append(element)
            else:
                # Sort current line by x-position and finalize
                current_line.sort(key=lambda e: e['x'])
                
                # Classify line type
                line_type = self._classify_receipt_line_type(current_line)
                
                # Add to lines
                lines.append({
                    'elements': current_line,
                    'y': current_y,
                    'type': line_type
                })
                
                # Start new line
                current_line = [element]
                current_y = element['y']
        
        # Add final line
        if current_line:
            current_line.sort(key=lambda e: e['x'])
            line_type = self._classify_receipt_line_type(current_line)
            lines.append({
                'elements': current_line,
                'y': current_y,
                'type': line_type
            })
        
        # Sort lines by y-position
        lines.sort(key=lambda line: line['y'])
        
        return lines
    
    def _classify_receipt_line_type(self, line_elements):
        """Classify the type of a receipt line"""
        if not line_elements:
            return 'empty'
        
        # Concatenate the text in this line
        line_text = ' '.join(e['text'] for e in line_elements)
        
        # Check for common receipt line patterns
        if re.search(r'(?:total|amount)[^:]*?[:=]\s*\$?[\d,.]+', line_text.lower()):
            return 'total'
            
        elif re.search(r'(?:subtotal|sub-total)[^:]*?[:=]\s*\$?[\d,.]+', line_text.lower()):
            return 'subtotal'
            
        elif re.search(r'(?:tax|vat|gst)[^:]*?[:=]\s*\$?[\d,.]+', line_text.lower()):
            return 'tax'
            
        elif re.search(r'(?:date|time)[^:]*?[:=]', line_text.lower()):
            return 'metadata'
            
        elif re.search(r'\d+\.\d{2}$', line_text) and len(line_elements) > 1:
            # Ends with a price and has multiple elements - likely an item
            return 'item'
            
        elif re.search(r'^(?:thank|thanks)', line_text.lower()):
            return 'footer'
            
        elif line_text.strip() and line_elements[0]['y'] < 200:
            # Non-empty text near top of receipt - likely header/merchant info
            return 'header'
            
        else:
            return 'content'
    
    def _format_columnar_receipt(self, lines, structure_info):
        """Format receipt with column alignment"""
        if not lines:
            return ""
            
        columns = structure_info['columns']
        result_lines = []
        
        # Determine if last column is likely a price column
        has_price_column = False
        price_column_index = -1
        
        if columns and isinstance(columns[-1], tuple) and columns[-1][0] == 'price_column':
            has_price_column = True
            price_column_index = len(columns) - 1
            price_column_pos = columns[-1][1]
        
        # Process each line
        for line in lines:
            elements = line['elements']
            line_type = line['type']
            
            if not elements:
                result_lines.append('')
                continue
            
            if line_type == 'item' and has_price_column:
                # For item lines with price column, format specially
                item_text = []
                price_text = ""
                
                for elem in elements:
                    # Check if element is in price column
                    if price_column_index >= 0 and abs(elem['x'] - price_column_pos) < 30:
                        price_text = elem['text']
                    else:
                        # Normalize potential bullet points
                        text = elem['text']
                        for old_bullet, new_bullet in self.item_bullet_fixes.items():
                            if text.startswith(old_bullet):
                                text = new_bullet + text[len(old_bullet):]
                                break
                        item_text.append(text)
                
                # Format with proper spacing
                item_part = ' '.join(item_text)
                
                if price_text:
                    # Right-align the price
                    spacing = max(1, 50 - len(item_part))
                    result_lines.append(f"{item_part}{' ' * spacing}{price_text}")
                else:
                    result_lines.append(item_part)
            
            elif line_type in ['total', 'subtotal', 'tax'] and has_price_column:
                # For total lines, format specially
                label_part = []
                value_part = ""
                
                for elem in elements:
                    if price_column_index >= 0 and abs(elem['x'] - price_column_pos) < 30:
                        value_part = elem['text']
                    else:
                        label_part.append(elem['text'])
                
                label_text = ' '.join(label_part)
                
                # Format with proper alignment
                if value_part:
                    spacing = max(1, 50 - len(label_text))
                    result_lines.append(f"{label_text}{' ' * spacing}{value_part}")
                else:
                    result_lines.append(label_text)
            
            else:
                # For other lines, just join with appropriate spacing
                formatted_line = self._format_line_elements(elements, columns)
                result_lines.append(formatted_line)
        
        # Join lines and apply final cleaning
        result_text = '\n'.join(result_lines)
        result_text = self._clean_receipt_text(result_text)
        
        return result_text
    
    def _format_simple_receipt(self, lines, structure_info):
        """Format simple receipt without column alignment"""
        if not lines:
            return ""
            
        result_lines = []
        
        for line in lines:
            elements = line['elements']
            
            if not elements:
                result_lines.append('')
                continue
            
            # Format elements with simple spacing
            formatted_line = ' '.join(elem['text'] for elem in elements)
            
            # Normalize bullet points for items
            for old_bullet, new_bullet in self.item_bullet_fixes.items():
                if formatted_line.startswith(old_bullet):
                    formatted_line = new_bullet + formatted_line[len(old_bullet):]
                    break
            
            result_lines.append(formatted_line)
        
        # Join lines and apply final cleaning
        result_text = '\n'.join(result_lines)
        result_text = self._clean_receipt_text(result_text)
        
        return result_text
    
    def _format_line_elements(self, elements, columns):
        """Format elements in a line with proper column alignment"""
        if not elements:
            return ""
        
        # For single element, return as-is
        if len(elements) == 1:
            return elements[0]['text']
        
        # For multiple elements, consider spacing
        result = [elements[0]['text']]
        
        for i in range(1, len(elements)):
            prev_elem = elements[i-1]
            curr_elem = elements[i]
            
            # Calculate spacing based on gap and columns
            gap = curr_elem['x'] - (prev_elem['x'] + prev_elem['w'])
            
            # Check if elements are in different columns
            in_different_columns = False
            for col_pos in columns:
                col_x = col_pos[1] if isinstance(col_pos, tuple) else col_pos
                if prev_elem['x'] < col_x and curr_elem['x'] >= col_x:
                    in_different_columns = True
                    break
            
            # Determine spacing
            if in_different_columns:
                result.append('  ')  # Double space for column separation
            elif gap > 30:
                result.append('  ')  # Double space for large gaps
            else:
                result.append(' ')   # Single space for normal gaps
            
            result.append(curr_elem['text'])
        
        return ''.join(result)
    
    def _clean_receipt_text(self, text):
        """Clean and normalize receipt text"""
        if not text:
            return ""
        
        # Apply symbol fixes
        for error, correction in self.receipt_symbol_fixes.items():
            text = text.replace(error, correction)
        
        # Fix common OCR errors in receipts
        text = re.sub(r'T[Oo][Tt][Aa][Ll]', 'Total', text, flags=re.IGNORECASE)
        text = re.sub(r'S[Uu][Bb][Tt][Oo][Tt][Aa][Ll]', 'Subtotal', text, flags=re.IGNORECASE)
        text = re.sub(r'([A-Z][a-z]+)T([A-Z][a-z]+)', r'\1 T\2', text)  # Fix missing space before "T"
        
        # Fix spacing around special symbols
        text = re.sub(r'(\d)([$/£€])', r'\1 \2', text)
        text = re.sub(r'([$/£€])(\d)', r'\1\2', text)
        
        # Fix common currency formatting
        text = re.sub(r'([$/£€])\s+(\d)', r'\1\2', text)  # Remove space between currency and number
        
        # Fix decimal values to ensure 2 decimal places
        text = re.sub(r'(\d+)\.(\d)(?!\d)', r'\1.\20', text)  # Add missing zero
        
        # Fix multiple spaces
        text = re.sub(r' {2,}', '  ', text)
        
        # Clean up line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _extract_receipt_info(self, text):
        """Extract structured information from receipt text"""
        info = {
            'total': None,
            'subtotal': None,
            'tax': None,
            'date': None,
            'time': None,
            'merchant': None,
            'items': []
        }
        
        # Extract total with multiple patterns
        total_patterns = [
            r'total\s*[:=]?\s*[$£€]?\s*([\d,.]+)',
            r'total\s*[$£€]?\s*([\d,.]+)',
            r'amount\s*[:=]?\s*[$£€]?\s*([\d,.]+)',
            r'sum\s*[:=]?\s*[$£€]?\s*([\d,.]+)',
            r'(?:to pay|payment)\s*[:=]?\s*[$£€]?\s*([\d,.]+)',
            r'(?<!\w)(?:sum|amt)(?!\w).*?[$£€]?\s*([\d,.]+)'
        ]
        
        for pattern in total_patterns:
            match = re.search(pattern, text.lower())
            if match:
                info['total'] = match.group(1).strip()
                break
        
        # Extract subtotal
        subtotal_patterns = [
            r'sub[\s-]*total\s*[:=]?\s*[$£€]?\s*([\d,.]+)',
            r'(?<!\w)(?:sub|subtot)(?!\w).*?[$£€]?\s*([\d,.]+)'
        ]
        
        for pattern in subtotal_patterns:
            match = re.search(pattern, text.lower())
            if match:
                info['subtotal'] = match.group(1).strip()
                break
        
        # Extract tax
        tax_patterns = [
            r'(?:tax|vat|gst)\s*[:=]?\s*[$£€]?\s*([\d,.]+)',
            r'(?<!\w)(?:hst|pst)(?!\w).*?[$£€]?\s*([\d,.]+)'
        ]
        
        for pattern in tax_patterns:
            match = re.search(pattern, text.lower())
            if match:
                info['tax'] = match.group(1).strip()
                break
        
        # Extract date
        date_patterns = [
            r'(?:date|dt)[^:]*?[:=]\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(?:date|dt)[^:]*?[:=]\s*(\d{2,4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})',
            r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                info['date'] = match.group(1).strip()
                break
        
        # Extract time
        time_patterns = [
            r'(?:time|tm)[^:]*?[:=]\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)',
            r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text)
            if match:
                info['time'] = match.group(1).strip()
                break
        
        # Extract merchant name (usually at top)
        lines = text.split('\n')
        if lines:
            # Look for merchant name in first few non-empty lines
            for line in (l.strip() for l in lines[:5] if l.strip()):
                # Skip lines with date, numbers, or other metadata
                if (not re.search(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', line) and
                    not re.search(r'(?:date|time|receipt|tel|phone|address|transaction)', line.lower()) and
                    len(line) > 3):
                    info['merchant'] = line
                    break
        
        # Extract items (lines with prices)
        # Look for lines with price pattern at the end, but not containing "total", "tax", etc.
        item_pattern = r'^(.*?)\s+[$£€]?\s*(\d+\.\d{2})$'
        
        # Skip header and total sections
        body_lines = []
        in_body = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Skip header lines
            if i < 5 and not in_body:
                continue
                
            # Start of body when we find an item pattern
            if not in_body and re.match(item_pattern, line):
                in_body = True
                
            # End of body when we find total indicators
            if in_body and re.search(r'(?:total|subtotal|tax|balance|sum|amount)', line.lower()):
                in_body = False
                
            if in_body:
                body_lines.append(line)
        
        # Process body lines for items
        items = []
        for line in body_lines:
            match = re.match(item_pattern, line)
            if match and len(match.group(1).strip()) > 1:  # Skip if item name too short
                item = {
                    'name': match.group(1).strip(),
                    'price': match.group(2).strip()
                }
                items.append(item)
        
        info['items'] = items
        
        return info
    
    def _error_result(self, error_message):
        """Return error result dictionary"""
        return {
            'text': '',
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'user': 'Tigran0000'
        }