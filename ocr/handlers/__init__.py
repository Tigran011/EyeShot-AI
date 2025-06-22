"""
OCR Document Handlers

This subpackage provides specialized document handlers for different document types.
Each handler is optimized for specific document characteristics and extraction requirements.

Available handlers:
- AcademicHandler: Scientific papers and formal documents
- BookHandler: Book pages with multi-column layouts
- CodeHandler: Source code with indentation preservation
- ReceiptHandler: Receipts and invoices
- TableHandler: Tables and structured data
- TitleHandler: Titles and headings
- FormHandler: Forms with fields and checkboxes
"""

# Import all handlers for direct availability
from .base_handler import DocumentHandler
from .academic_handler import AcademicHandler
from .book_handler import BookHandler
from .code_handler import CodeHandler
from .receipt_handler import ReceiptHandler
from .table_handler import TableHandler
from .title_handler import TitleHandler
from .form_handler import FormHandler

# Dictionary mapping document types to their handlers
# Useful for automatic handler selection
DOCUMENT_HANDLERS = {
    'academic': AcademicHandler,
    'book': BookHandler,
    'code': CodeHandler,
    'receipt': ReceiptHandler,
    'table': TableHandler,
    'title': TitleHandler,
    'form': FormHandler,
    'standard': DocumentHandler,
}
