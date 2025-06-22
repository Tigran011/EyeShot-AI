"""
EyeShot AI - Text Processor
Handles post-processing of extracted text to improve quality and formatting
Last updated: 2025-06-20 10:02:38 UTC
Author: Tigran0000
"""

import re
from typing import Dict, List, Optional, Tuple, Any

class TextProcessor:
    """
    Processes extracted text to improve quality, fix common OCR errors, 
    and preserve document structure.
    """
    
    def __init__(self):
        """Initialize text processor with default settings"""
        # Common OCR error patterns and their corrections
        self.error_patterns = {
            r'[\u2022\u2023\u2043\u204C\u204D\u2219\u25D8\u25E6\u2619\u2765\u2767\u29BE\u29BF]': '• ',  # Various bullet characters
            r'([a-zA-Z])•': r'\1e •',  # Fix common "e" to bullet error (Phas• -> Phase •)
            r'•([a-zA-Z])': r'• \1',  # Add space after bullet if missing
            r'\s{2,}': ' ',  # Normalize multiple spaces to single space
            r'([a-zA-Z0-9])\n([a-z])': r'\1 \2',  # Join broken lines, but only if second line starts with lowercase
            r'(\d+)\.(\d+)': r'\1.\2',  # Fix decimal points without space
            r'([a-zA-Z])(\d)': r'\1 \2',  # Add space between letter and number if missing
            r'(\d)([a-zA-Z])': r'\1 \2',  # Add space between number and letter if missing
        }
        
        # Patterns to protect during general cleanup
        self.protect_patterns = [
            r'•\s+[^\n]+',  # Bullet points
            r'\d+\.\s+[^\n]+',  # Numbered list items
            r'[A-Z]\.\s+[^\n]+',  # Letter list items
            r'[a-z]\)\s+[^\n]+',  # Letter with parenthesis list items
        ]
        
        # Settings
        self.preserve_line_breaks = True
        self.merge_paragraphs = False
        self.fix_common_errors = True
        self.preserve_bullet_points = True
        self.preserve_indentation = True
        
    def clean_extracted_text(self, text: str, document_type: str = 'standard') -> str:
        """
        Clean and format extracted text based on document type
        
        Args:
            text: Raw extracted text
            document_type: Type of document for specialized processing
            
        Returns:
            Cleaned and formatted text
        """
        if not text:
            return ""
            
        # Adapt processing based on document type
        if document_type == 'code':
            self.preserve_line_breaks = True
            self.merge_paragraphs = False
            self.preserve_indentation = True
        elif document_type == 'receipt':
            self.preserve_line_breaks = True
            self.merge_paragraphs = False
            self.preserve_indentation = False
        elif document_type in ['academic', 'book']:
            self.preserve_line_breaks = False
            self.merge_paragraphs = True
            self.preserve_indentation = False
        else:  # standard, form, etc.
            self.preserve_line_breaks = True
            self.merge_paragraphs = False
            self.preserve_indentation = True
            
        # Apply text cleaning steps
        cleaned_text = text
        
        # Pre-process: normalize line endings
        cleaned_text = cleaned_text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Fix bullet points and lists
        if self.preserve_bullet_points:
            # First protect existing well-formed bullets
            protected_sections = {}
            for i, pattern in enumerate(self.protect_patterns):
                for match in re.finditer(pattern, cleaned_text):
                    placeholder = f"__PROTECTED_{i}_{len(protected_sections)}__"
                    protected_sections[placeholder] = match.group(0)
                    cleaned_text = cleaned_text.replace(match.group(0), placeholder, 1)
                    
            # Fix common issues with bullet points
            for pattern, replacement in self.error_patterns.items():
                cleaned_text = re.sub(pattern, replacement, cleaned_text)
                
            # Restore protected sections
            for placeholder, original in protected_sections.items():
                cleaned_text = cleaned_text.replace(placeholder, original)
        
        # Handle paragraph merging
        if not self.preserve_line_breaks and self.merge_paragraphs:
            # Convert single line breaks to spaces for paragraph continuity
            # but preserve paragraph breaks (double line breaks)
            cleaned_text = re.sub(r'([^\n])\n([^\n])', r'\1 \2', cleaned_text)
            
        # Fix spacing after periods and common punctuation
        cleaned_text = re.sub(r'([.!?])([A-Z])', r'\1 \2', cleaned_text)
        
        # Remove excessive spacing
        cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
        
        # Final formatting based on document type
        if document_type == 'title':
            # Title text often benefits from being on a single line
            cleaned_text = cleaned_text.replace('\n', ' ')
            cleaned_text = re.sub(r' {2,}', ' ', cleaned_text).strip()
            
        elif document_type == 'form':
            # For forms, preserve line structure is essential
            pass
            
        elif document_type == 'table':
            # For tables, ensure columns are properly aligned
            # This could be enhanced with more sophisticated table structure preservation
            cleaned_text = self._align_table_columns(cleaned_text)
        
        return cleaned_text
    
    def _align_table_columns(self, text: str) -> str:
        """Attempt to align table columns for better readability"""
        lines = text.split('\n')
        if not lines:
            return text
            
        # Simple column alignment heuristic
        max_line_length = max(len(line) for line in lines)
        columns = []
        
        # Try to identify column positions by looking for consistent spacing
        spaces = [0] * max_line_length
        for line in lines:
            for i, char in enumerate(line):
                if char.isspace():
                    spaces[i] += 1
                    
        # Find potential column boundaries (spaces that occur in most lines)
        threshold = len(lines) * 0.6  # 60% of lines should have space at this position
        potential_columns = [i for i, count in enumerate(spaces) if count >= threshold]
        
        # If no clear columns found, or too many, don't try to align
        if not potential_columns or len(potential_columns) > 10:
            return text
            
        # Identify distinct columns (not too close to each other)
        column_positions = []
        last_pos = -5  # Start with an offset
        for pos in potential_columns:
            if pos - last_pos >= 3:  # Minimum 3 chars between columns
                column_positions.append(pos)
                last_pos = pos
                
        # Align text based on columns
        if column_positions:
            aligned_lines = []
            for line in lines:
                new_line = list(line)
                for col_pos in column_positions:
                    if col_pos < len(new_line):
                        if not new_line[col_pos].isspace():
                            # Insert space to align columns
                            new_line.insert(col_pos, ' ')
                aligned_lines.append(''.join(new_line))
            return '\n'.join(aligned_lines)
            
        return text
    
    def restore_list_formatting(self, text: str) -> str:
        """
        Specially process lists to ensure proper formatting of bullet points and numbering
        """
        if not text:
            return ""
            
        processed_text = text
        
        # Fix bullet point formatting
        # Format: • Item text
        processed_text = re.sub(r'([^\s])•', r'\1 •', processed_text)  # Add space before bullet if missing
        processed_text = re.sub(r'•([^\s])', r'• \1', processed_text)  # Add space after bullet if missing
        
        # Fix numbered list formatting
        # Format: 1. Item text
        processed_text = re.sub(r'(\d+)\.([^\s])', r'\1. \2', processed_text)  # Add space after period if missing
        
        # Fix lettered list formatting
        # Format: a) Item text or A. Item text
        processed_text = re.sub(r'([a-zA-Z])\)([^\s])', r'\1) \2', processed_text)
        processed_text = re.sub(r'([A-Z])\.([^\s])', r'\1. \2', processed_text)
        
        return processed_text
    
    def fix_common_ocr_errors(self, text: str) -> str:
        """Correct common OCR errors based on known patterns"""
        if not text or not self.fix_common_errors:
            return text
            
        corrected_text = text
        
        # Common OCR errors and their corrections
        error_corrections = {
            # Character substitutions
            'l1': 'll',  # lowercase L to one
            '0': 'O',  # zero to capital O in certain contexts
            '1': 'I',  # one to capital I in certain contexts
            '5': 'S',  # five to capital S in certain contexts
            
            # Word corrections (common OCR misreads)
            'cornputer': 'computer',
            'systern': 'system',
            'frorn': 'from',
            'rnodel': 'model',
            'docurnent': 'document',
            'rnethod': 'method',
        }
        
        # Apply context-sensitive corrections
        for error, correction in error_corrections.items():
            # Only correct in appropriate contexts to avoid false positives
            if error in ['0', '1', '5']:
                # For number/letter confusions, only correct within words
                pattern = r'([a-zA-Z])' + re.escape(error) + r'([a-zA-Z])'
                replace = r'\1' + correction + r'\2'
                corrected_text = re.sub(pattern, replace, corrected_text)
            else:
                # For other errors, use simple replacement
                corrected_text = corrected_text.replace(error, correction)
                
        return corrected_text
    
    def preserve_document_structure(self, text: str, structure_info: Dict = None) -> str:
        """
        Preserve document structure based on structural hints
        
        Args:
            text: Processed text
            structure_info: Dictionary with structure information
            
        Returns:
            Text with preserved structure
        """
        if not text:
            return ""
            
        if not structure_info:
            return text
            
        structured_text = text
        
        # Handle multi-column layout
        columns = structure_info.get('columns', 1)
        if columns > 1 and 'blocks' in structure_info:
            structured_text = self._reconstruct_columns(text, structure_info)
            
        # Handle indentation
        if self.preserve_indentation and 'indentation_levels' in structure_info:
            structured_text = self._restore_indentation(structured_text, structure_info)
            
        # Handle paragraphs
        if 'paragraphs' in structure_info:
            structured_text = self._format_paragraphs(structured_text, structure_info)
            
        return structured_text
    
    def _reconstruct_columns(self, text: str, structure_info: Dict) -> str:
        """Reconstruct multi-column layout"""
        # This would require the original block positions from the OCR
        # and would reorder text blocks based on column layout
        # For now, we'll return the original text
        return text
    
    def _restore_indentation(self, text: str, structure_info: Dict) -> str:
        """Restore indentation based on structure information"""
        indentation_levels = structure_info.get('indentation_levels', [])
        if not indentation_levels:
            return text
            
        lines = text.split('\n')
        indented_lines = []
        
        # Simple heuristic: lines starting with bullet points or numbers get indented
        for line in lines:
            stripped = line.lstrip()
            if re.match(r'^[•\*\-]\s', stripped) or re.match(r'^\d+\.\s', stripped):
                # Find appropriate indentation level
                original_indent = len(line) - len(stripped)
                target_indent = 0
                
                # Find closest indentation level
                if original_indent > 0 and indentation_levels:
                    closest = min(indentation_levels, key=lambda x: abs(x - original_indent))
                    if closest > 2:  # Only use significant indentation
                        target_indent = closest
                
                # Apply indentation
                indented_lines.append(' ' * target_indent + stripped)
            else:
                indented_lines.append(line)
                
        return '\n'.join(indented_lines)
    
    def _format_paragraphs(self, text: str, structure_info: Dict) -> str:
        """Format paragraphs based on structure information"""
        paragraphs = structure_info.get('paragraphs', [])
        if not paragraphs:
            return text
            
        # This would require more complex logic to reconstruct paragraphs
        # For now, we'll ensure proper spacing between paragraphs
        lines = text.split('\n')
        formatted_lines = []
        in_paragraph = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Simple heuristic for paragraph breaks
            is_paragraph_start = False
            is_paragraph_end = False
            
            if stripped:
                if not in_paragraph:
                    is_paragraph_start = True
                    in_paragraph = True
                    
                # Check if this is the end of a paragraph
                if i < len(lines) - 1:
                    next_stripped = lines[i + 1].strip()
                    if not next_stripped:
                        is_paragraph_end = True
                        in_paragraph = False
            else:
                in_paragraph = False
                
            # Format based on paragraph position
            if is_paragraph_start and formatted_lines:
                # Add extra line before paragraph if not already there
                if formatted_lines[-1].strip():
                    formatted_lines.append('')
                    
            formatted_lines.append(line)
            
            if is_paragraph_end:
                # No need to add extra line as the next line is already empty
                pass
                
        return '\n'.join(formatted_lines)