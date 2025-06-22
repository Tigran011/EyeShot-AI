# EyeShot AI

Smart OCR text extraction app using AI-powered document handlers.

## Features

- Book, academic, code, table, receipt, and form handler support
- Advanced layout analysis (columns, drop caps, tables, etc.)
- PyQt5 UI for easy file import and result browsing
- PDF page conversion and multi-page navigation
- **AI-integrated OCR**: Uses EasyOCR + PyTorch for advanced text recognition, including support for challenging fonts, handwriting, and noisy scans

## Quick Start (for Windows)

### 1. **Install Tesseract OCR**

- Download and install the latest version of Tesseract from [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki).
- Add the Tesseract install folder (e.g., `C:\Program Files\Tesseract-OCR`) to your system PATH.
- Verify the installation by opening a new Command Prompt and running:
  ```cmd
  tesseract --version
  ```

### 2. **Install Python dependencies**

Open a terminal (such as Command Prompt or PowerShell), navigate to the project folder, and run:
```bash
pip install -r requirements.txt
```

### 3. **Launch the app**

```bash
python main.py
```

---

## Directory Structure

- `core/` - File handling and app core logic
- `ocr/` - OCR engine, handlers, image/text processing
- `ui/` - PyQt5 UI components and main window
- `converters/` - PDF/image conversion helpers
- `exporters/` - Export utilities

---

## AI Integration

EyeShot AI uses state-of-the-art deep learning OCR engines (EasyOCR + PyTorch) alongside classic Tesseract OCR for robust, accurate text extraction—even with low-quality scans, handwriting, or stylized fonts. No GPU required; works on most modern Windows PCs.

---

## License

MIT

---

## Requirements

See [requirements.txt](./requirements.txt) for all dependencies.
