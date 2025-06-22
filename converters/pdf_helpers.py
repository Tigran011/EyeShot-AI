"""
EyeShot AI - PDF Helper Functions
Provides simplified interfaces to PDF conversion functionality
Last updated: 2025-06-20 10:24:15 UTC
Author: Tigran0000
"""

import os
from PIL import Image
from typing import Dict, Optional, Any, Union

def create_enhanced_pdf_image(pdf_path: str, page_number: int, dpi: int = 300, 
                            document_type: str = None, optimize_for_ocr: bool = True) -> Dict[str, Any]:
    """
    Create an enhanced PDF image specifically optimized for OCR.
    This is a bridge function that imports the converter only when needed.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number (0-based)
        dpi: Resolution in DPI
        document_type: Type of document for specialized processing
                      (None = auto-detect)
        optimize_for_ocr: Apply specialized OCR optimizations
        
    Returns:
        Dict containing image and structure information
    """
    # Only import converter when function is called to avoid circular imports
    from converters.pdf_to_image import PDFConverter, PDFAnalyzer
    
    try:
        # Create converter instance
        converter = PDFConverter()
        
        # Load PDF
        result = converter.load_pdf(pdf_path)
        if not result['success']:
            print(f"Failed to load PDF: {result.get('error')}")
            return {'success': False, 'error': result.get('error')}
        
        # Auto-detect document type if not specified
        detected_type = None
        if not document_type:
            try:
                analysis = PDFAnalyzer.analyze_pdf_for_ocr(pdf_path)
                if analysis['success']:
                    detected_type = analysis.get('document_type', 'standard')
                    document_type = detected_type
            except Exception as e:
                print(f"Document type detection failed (using 'standard'): {e}")
                document_type = 'standard'
            
        # Get enhanced image with document type optimization
        image = converter.convert_page_to_image_enhanced(
            page_number, 
            optimize_for_ocr=optimize_for_ocr,
            document_type=document_type
        )
        
        if not image:
            print("Failed to create enhanced image")
            return {'success': False, 'error': 'Failed to create enhanced image'}
            
        # Get structure information
        structure = converter.detect_page_structure(page_number)
        structure_hints = converter.extract_structure_hints(page_number)
        
        # Add paragraph analysis if the method exists
        try:
            # Check if the method exists before calling it
            if hasattr(converter, 'analyze_paragraph_structure'):
                paragraph_structure = converter.analyze_paragraph_structure(page_number)
                
                # Merge paragraph data into structure hints
                if paragraph_structure and paragraph_structure.get('success'):
                    for key in ['paragraphs', 'heading_blocks', 'line_spacing_patterns',
                              'paragraph_first_line_indents', 'hyphenated_line_ends']:
                        if key in paragraph_structure:
                            structure_hints[key] = paragraph_structure[key]
                    
                    # Add paragraph count for convenience
                    if 'paragraphs' in paragraph_structure:
                        structure_hints['paragraph_count'] = len(paragraph_structure['paragraphs'])
            else:
                # If method doesn't exist, add basic paragraph data
                structure_hints['paragraph_count'] = 0
                structure_hints['paragraphs'] = []
        except Exception as e:
            print(f"Paragraph analysis failed (continuing): {e}")
            # Add empty paragraph data
            structure_hints['paragraph_count'] = 0
            structure_hints['paragraphs'] = []
        
        # Clean up
        converter.close_pdf()
        
        return {
            'image': image,
            'structure': structure,
            'structure_hints': structure_hints,
            'document_type': document_type,
            'detected_type': detected_type,
            'page_number': page_number,
            'total_pages': result['page_count'],
            'success': True
        }
    except Exception as e:
        import traceback
        print(f"Error creating enhanced PDF image: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}