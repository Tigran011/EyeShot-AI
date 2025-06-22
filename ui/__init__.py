def __init__(self):
    super().__init__()
    
    # Initialize file handler, OCR engine, and PDF converter
    self.file_handler = FileHandler()
    self.ocr_engine = OCREngine()
    
    print("🔍 DEBUG: Initializing PDF converter...")
    try:
        self.pdf_converter = PDFConverter()
        print("✅ DEBUG: PDF converter initialized successfully")
    except Exception as e:
        print(f"❌ DEBUG: Failed to initialize PDF converter: {e}")
        self.pdf_converter = None
    
    # Test OCR installation on startup
    self.test_ocr_on_startup()
    
    # ... rest of __init__ code ...
    
    # PDF-specific state
    self.current_page = 0
    self.total_pages = 0
    self.pdf_images = {}  # Cache converted pages
    
    self.statusBar().showMessage("Ready - Open an image or PDF to get started")
