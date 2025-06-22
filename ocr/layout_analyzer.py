"""
EyeShot AI - Layout Analyzer
Analyzes document layout to improve OCR structure preservation
Last updated: 2025-06-20 10:29:22 UTC
Author: Tigran0000
"""

import numpy as np
import cv2
import json
import re
from typing import Dict, List, Tuple, Any, Union
from PIL import Image

# This will be used conditionally
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pass

class LayoutAnalyzer:
    """Analyzes document layout for improved OCR"""
    
    def __init__(self):
        """Initialize the layout analyzer"""
        self.debug_mode = False
    
    def analyze_layout(self, image: Union[np.ndarray, Image.Image]) -> Dict:
        """
        Analyze document layout to identify different regions
        
        Args:
            image: PIL Image or numpy array to analyze
            
        Returns:
            Dictionary with layout information
        """
        # First convert PIL Image to numpy if needed
        try:
            if isinstance(image, Image.Image):
                # Convert PIL Image to numpy array
                image_array = np.array(image)
            else:
                # Already numpy array
                image_array = image
                
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Initialize layout info
            layout_info = {
                'regions': []
            }
            
            # Get image dimensions
            height, width = gray.shape
            layout_info['width'] = width
            layout_info['height'] = height
            
            # Apply adaptive threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze each contour
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip very small regions
                if w < 20 or h < 20:
                    continue
                
                # Calculate region size relative to image
                rel_size = (w * h) / (width * height)
                
                # Skip tiny regions (noise)
                if rel_size < 0.001:
                    continue
                
                # Determine region type based on properties
                region_type = self._classify_region_type(gray[y:y+h, x:x+w], w, h)
                
                # Add region to layout info
                layout_info['regions'].append({
                    'type': region_type,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'confidence': 0.8  # Default confidence
                })
            
            # Add column detection
            columns = self._detect_columns(gray)
            layout_info['columns'] = columns
            
            # Add paragraph detection
            paragraphs = self._detect_paragraphs(gray)
            layout_info['paragraphs'] = paragraphs
            
            # Add layout type
            layout_info['layout_type'] = self._determine_layout_type(columns, paragraphs)
            
            # Add other flags
            layout_info['has_table'] = any(region['type'] == 'table' for region in layout_info['regions'])
            layout_info['has_list'] = self._detect_lists(gray)
            layout_info['has_heading'] = any(region['type'] == 'title' for region in layout_info['regions'])
            
            return layout_info
            
        except Exception as e:
            # Return empty layout on error
            if self.debug_mode:
                print(f"Layout analysis error: {e}")
                
            # Get basic dimensions even on error
            if isinstance(image, Image.Image):
                width, height = image.size
            else:
                try:
                    height, width = image.shape[:2]
                except:
                    width, height = 0, 0
                    
            return {
                'regions': [],
                'width': width,
                'height': height,
                'columns': 1,
                'paragraphs': 1,
                'has_table': False,
                'has_list': False,
                'has_heading': False,
                'layout_type': 'simple'
            }
    
    def _classify_region_type(self, region: np.ndarray, width: int, height: int) -> str:
        """Classify region type based on visual characteristics"""
        
        # Calculate aspect ratio
        aspect_ratio = width / max(height, 1)  # Avoid division by zero
        
        # Calculate white pixel ratio
        white_pixels = np.sum(region > 200)
        total_pixels = width * height
        white_ratio = white_pixels / max(total_pixels, 1)
        
        # Calculate edge density
        edges = cv2.Canny(region, 100, 200)
        edge_pixels = np.sum(edges > 0)
        edge_density = edge_pixels / max(total_pixels, 1)
        
        # Classification logic
        if white_ratio > 0.9:
            return 'blank'
        elif edge_density > 0.1 and width > 100 and height > 100:
            # Check if it's likely a table
            h_lines, v_lines = self._detect_table_lines(region)
            if h_lines > 2 and v_lines > 2:
                return 'table'
        
        # Check if it's likely a title
        if aspect_ratio > 2 and height < 80 and edge_density < 0.05:
            return 'title'
        
        # Check if it's likely an image
        if edge_density > 0.2 and white_ratio < 0.5:
            return 'image'
            
        # Default to text
        return 'text'
    
    def _detect_table_lines(self, img: np.ndarray) -> tuple:
        """Detect horizontal and vertical lines for table detection"""
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Get dimensions
        height, width = binary.shape
        
        # Define line detection kernels
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width//10, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height//10))
        
        # Apply morphology operations
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Count horizontal and vertical lines
        h_lines = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        v_lines = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        return len(h_lines), len(v_lines)
    
    def _detect_columns(self, image_array: np.ndarray) -> int:
        """
        Detect number of text columns in document
        
        Args:
            image_array: Numpy array of image
            
        Returns:
            Number of columns detected
        """
        try:
            # Simple column detection based on vertical intensity projection
            # Binary threshold to isolate text
            _, binary = cv2.threshold(image_array, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Sum pixels vertically to find text density
            vertical_projection = np.sum(binary, axis=0)
            
            # Smooth projection to reduce noise
            kernel_size = max(5, image_array.shape[1] // 100)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Must be odd
            
            # Apply smoothing
            smoothed = np.zeros_like(vertical_projection, dtype=float)
            
            # Manual smoothing with moving average
            half_window = kernel_size // 2
            for i in range(len(vertical_projection)):
                start = max(0, i - half_window)
                end = min(len(vertical_projection), i + half_window + 1)
                smoothed[i] = np.mean(vertical_projection[start:end])
            
            # Normalize for visualization and analysis
            if np.max(smoothed) > 0:
                normalized = smoothed / np.max(smoothed)
            else:
                return 1
            
            # Calculate average density
            avg_density = np.mean(normalized)
            
            # Look specifically for a valley in the middle (common in two-column books)
            middle = image_array.shape[1] // 2
            middle_region_start = max(0, middle - image_array.shape[1]//8)
            middle_region_end = min(len(normalized), middle + image_array.shape[1]//8)
            middle_region = normalized[middle_region_start:middle_region_end]
            
            # If there's a significant drop in the middle, it's likely a two-column layout
            if len(middle_region) > 0 and np.min(middle_region) < avg_density * 0.5:
                return 2
            
            # Check for more complex multi-column layouts
            valleys = []
            min_valley_width = image_array.shape[1] * 0.02  # Min width of a valley (2% of image width)
            min_valley_drop = avg_density * 0.4  # Valley must be at least 40% lower than average
            
            i = 0
            while i < len(normalized):
                if normalized[i] < avg_density - min_valley_drop:
                    valley_start = i
                    # Find where valley ends
                    while i < len(normalized) and normalized[i] < avg_density - min_valley_drop:
                        i += 1
                    valley_end = i
                    
                    # Check if valley is wide enough to be a column separator
                    if valley_end - valley_start >= min_valley_width:
                        valleys.append((valley_start, valley_end))
                else:
                    i += 1
            
            # If we found valid valleys, count columns
            if valleys:
                return len(valleys) + 1
                
            # Default to single column
            return 1
                
        except Exception as e:
            if self.debug_mode:
                print(f"Column detection error: {e}")
            return 1
    
    def _detect_paragraphs(self, image_array: np.ndarray) -> int:
        """
        Estimate number of paragraphs in document
        
        Args:
            image_array: Numpy array of image
            
        Returns:
            Estimated number of paragraphs
        """
        try:
            # Look for horizontal gaps that might indicate paragraph breaks
            threshold = np.mean(image_array)
            binary = image_array < threshold
            
            # Sum horizontally to find rows with text
            horizontal_projection = np.sum(binary, axis=1)
            
            # Find empty rows (potential paragraph separators)
            empty_threshold = np.mean(horizontal_projection) * 0.2
            empty_rows = horizontal_projection < empty_threshold
            
            # Count transitions from empty to non-empty as paragraph starts
            paragraphs = 0
            in_empty = True
            
            for is_empty in empty_rows:
                if in_empty and not is_empty:
                    # Transition from empty to non-empty - paragraph start
                    paragraphs += 1
                    in_empty = False
                elif not in_empty and is_empty:
                    # Transition from non-empty to empty
                    in_empty = True
            
            # Ensure at least one paragraph is detected
            return max(1, paragraphs)
                
        except Exception as e:
            if self.debug_mode:
                print(f"Paragraph detection error: {e}")
            return 1
    
    def _detect_lists(self, image_array: np.ndarray) -> bool:
        """Simple list detection"""
        # TODO: Implement more sophisticated list detection
        # For now, this is very basic
        return False
    
    def _determine_layout_type(self, columns: int, paragraphs: int) -> str:
        """Determine layout type based on columns and paragraphs"""
        if columns > 1:
            return "multi_column"
        elif paragraphs > 5:
            return "complex"
        else:
            return "simple"
    
    def extract_table_structure(self, image: Union[np.ndarray, Image.Image]) -> Dict:
        """Extract table structure from image"""
        
        # First convert PIL Image to numpy if needed
        try:
            if isinstance(image, Image.Image):
                # Convert PIL Image to numpy array
                image_array = np.array(image)
            else:
                # Already numpy array
                image_array = image
                
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Initialize table info
            table_data = {
                'rows': 0,
                'columns': 0,
                'cells': [],
                'confidence': 0.0
            }
            
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(
                gray, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                11, 
                2
            )
            
            # Detect lines
            horizontal, vertical = self._detect_table_grid(binary)
            
            # Find intersections to identify cells
            intersections = cv2.bitwise_and(horizontal, vertical)
            
            # Find contours of intersections
            contours, _ = cv2.findContours(intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort intersection points
            points = []
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append((cx, cy))
            
            if not points:
                return table_data
            
            # Cluster points to find unique rows and columns
            row_positions = self._cluster_points([p[1] for p in points])
            col_positions = self._cluster_points([p[0] for p in points])
            
            # Update table info
            table_data['rows'] = len(row_positions) - 1
            table_data['columns'] = len(col_positions) - 1
            
            # Extract cell content
            if table_data['rows'] > 0 and table_data['columns'] > 0:
                total_confidence = 0
                cell_count = 0
                
                for r in range(len(row_positions) - 1):
                    for c in range(len(col_positions) - 1):
                        # Define cell boundaries
                        top = row_positions[r]
                        bottom = row_positions[r+1]
                        left = col_positions[c]
                        right = col_positions[c+1]
                        
                        # Extract cell image
                        cell_img = gray[top:bottom, left:right]
                        
                        # Skip empty cells
                        if cell_img.size == 0:
                            continue
                            
                        # Apply local threshold to remove grid lines
                        _, cell_binary = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        # OCR the cell content (basic version)
                        cell_text = ""
                        cell_confidence = 0
                        
                        if TESSERACT_AVAILABLE:
                            try:
                                # Simple OCR for the cell
                                cell_text = pytesseract.image_to_string(cell_binary, config='--psm 6').strip()
                                
                                # Get confidence
                                data = pytesseract.image_to_data(cell_binary, output_type=pytesseract.Output.DICT)
                                confidences = [int(c) for c in data['conf'] if int(c) > 0]
                                cell_confidence = sum(confidences) / len(confidences) if confidences else 0
                            except:
                                cell_text = ""
                                cell_confidence = 0
                        
                        # Add cell to table data
                        if cell_text:
                            table_data['cells'].append({
                                'row': r,
                                'col': c,
                                'text': cell_text,
                                'confidence': cell_confidence,
                                'bbox': (left, top, right, bottom)
                            })
                            
                            total_confidence += cell_confidence
                            cell_count += 1
                
                # Calculate overall confidence
                if cell_count > 0:
                    table_data['confidence'] = total_confidence / cell_count
                
                # Generate markdown table
                table_data['markdown_table'] = self._generate_markdown_table(table_data)
                
                # Generate JSON table
                table_data['json_table'] = json.dumps(self._generate_json_table(table_data))
            
            return table_data
            
        except Exception as e:
            if self.debug_mode:
                print(f"Table extraction error: {e}")
            return {'rows': 0, 'columns': 0, 'cells': []}
    
    def _detect_table_grid(self, binary: np.ndarray) -> tuple:
        """Detect horizontal and vertical lines in a table"""
        
        # Get image dimensions
        height, width = binary.shape
        
        # Define horizontal kernel
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 30, 1))
        
        # Detect horizontal lines
        horizontal_temp = cv2.erode(binary, horizontal_kernel)
        horizontal = cv2.dilate(horizontal_temp, horizontal_kernel)
        
        # Define vertical kernel
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 30))
        
        # Detect vertical lines
        vertical_temp = cv2.erode(binary, vertical_kernel)
        vertical = cv2.dilate(vertical_temp, vertical_kernel)
        
        return horizontal, vertical
    
    def _cluster_points(self, points: list, threshold: int = 10) -> list:
        """Cluster points that are close to each other"""
        
        if not points:
            return []
            
        # Sort points
        points = sorted(points)
        
        # Initialize clusters with first point
        clusters = [points[0]]
        
        # Cluster points
        for point in points[1:]:
            if point - clusters[-1] < threshold:
                # Update cluster center
                clusters[-1] = (clusters[-1] + point) // 2
            else:
                # Start new cluster
                clusters.append(point)
        
        return clusters
    
    def _generate_markdown_table(self, table_data: Dict) -> str:
        """Generate a markdown table from extracted table data"""
        
        rows = table_data['rows']
        cols = table_data['columns']
        
        if rows <= 0 or cols <= 0:
            return ""
        
        # Create empty table
        table = [["" for _ in range(cols)] for _ in range(rows)]
        
        # Fill in cell content
        for cell in table_data['cells']:
            r, c = cell['row'], cell['col']
            if 0 <= r < rows and 0 <= c < cols:
                table[r][c] = cell['text']
        
        # Build markdown
        markdown = []
        
        # Header row
        markdown.append("| " + " | ".join(table[0]) + " |")
        
        # Separator row
        markdown.append("| " + " | ".join(["---"] * cols) + " |")
        
        # Data rows
        for row in table[1:]:
            markdown.append("| " + " | ".join(row) + " |")
        
        return "\n".join(markdown)
    
    def _generate_json_table(self, table_data: Dict) -> Dict:
        """Generate a JSON representation of the table"""
        
        rows = table_data['rows']
        cols = table_data['columns']
        
        if rows <= 0 or cols <= 0:
            return {}
        
        # Create empty table
        table = [["" for _ in range(cols)] for _ in range(rows)]
        
        # Fill in cell content
        for cell in table_data['cells']:
            r, c = cell['row'], cell['col']
            if 0 <= r < rows and 0 <= c < cols:
                table[r][c] = cell['text']
        
        # Try to identify headers
        headers = table[0]
        data = []
        
        # Generate JSON representation
        for row_idx in range(1, rows):
            row_data = {}
            for col_idx in range(cols):
                header = headers[col_idx] if headers[col_idx] else f"column_{col_idx}"
                row_data[header] = table[row_idx][col_idx]
            data.append(row_data)
        
        return {'headers': headers, 'data': data}
    
    def extract_form_structure(self, image: Union[np.ndarray, Image.Image]) -> Dict:
        """Extract form structure from image"""
        
        # First convert PIL Image to numpy if needed
        try:
            if isinstance(image, Image.Image):
                # Convert PIL Image to numpy array
                image_array = np.array(image)
            else:
                # Already numpy array
                image_array = image
                
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Initialize form data
            form_data = {
                'fields': {},
                'confidence': 0.0
            }
            
            # Apply threshold
            binary = cv2.adaptiveThreshold(
                gray, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                11, 
                2
            )
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours to find likely form fields
            field_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip small contours
                if w < 50 or h < 15:
                    continue
                
                # Look for rectangles with aspect ratio typical of form fields
                aspect_ratio = w / h
                if 2 <= aspect_ratio <= 10:
                    field_contours.append((x, y, w, h))
            
            # Sort by vertical position
            field_contours.sort(key=lambda c: c[1])
            
            # Extract field text
            total_confidence = 0
            field_count = 0
            
            for i, (x, y, w, h) in enumerate(field_contours):
                # Expand region slightly to capture surrounding text
                expanded_x = max(0, x - 100)
                expanded_w = min(gray.shape[1] - expanded_x, w + 200)
                expanded_y = max(0, y - 20)
                expanded_h = min(gray.shape[0] - expanded_y, h + 40)
                
                # Extract expanded region
                field_region = gray[expanded_y:expanded_y+expanded_h, expanded_x:expanded_x+expanded_w]
                
                # OCR the field region
                if TESSERACT_AVAILABLE:
                    try:
                        # Extract text
                        text = pytesseract.image_to_string(field_region).strip()
                        
                        # Get confidence
                        data = pytesseract.image_to_data(field_region, output_type=pytesseract.Output.DICT)
                        confidences = [int(c) for c in data['conf'] if int(c) > 0]
                        field_confidence = sum(confidences) / len(confidences) if confidences else 0
                        
                        # Try to identify field name and value
                        match = re.search(r'([A-Za-z][A-Za-z\s]+)[:\-]\s*(.*)', text)
                        if match:
                            field_name = match.group(1).strip()
                            field_value = match.group(2).strip()
                            
                            if field_name and field_value:
                                form_data['fields'][field_name] = field_value
                                total_confidence += field_confidence
                                field_count += 1
                    except:
                        pass
            
            # Calculate overall confidence
            if field_count > 0:
                form_data['confidence'] = total_confidence / field_count
            
            return form_data
            
        except Exception as e:
            if self.debug_mode:
                print(f"Form extraction error: {e}")
            return {'fields': {}}