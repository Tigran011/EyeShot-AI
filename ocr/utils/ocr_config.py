"""
EyeShot AI - Tesseract Configuration Utility
Ensures proper Tesseract setup for PDF processing
"""

import os
import sys
import platform
import shutil
from pathlib import Path

def configure_tesseract():
    """
    Configure Tesseract with proper environment variables and language data
    """
    try:
        import pytesseract
        
        # 1. Find tesseract executable
        tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
        
        if not tesseract_cmd or not os.path.exists(tesseract_cmd):
            # Try to find it in common locations
            tesseract_cmd = find_tesseract_executable()
            
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                print(f"✅ Found Tesseract executable at: {tesseract_cmd}")
            else:
                print("❌ Tesseract executable not found")
                return False
        
        # 2. Find and set tessdata directory
        tessdata_dir = find_tessdata_dir(tesseract_cmd)
        
        if tessdata_dir:
            # Normalize path with proper OS-specific separators
            tessdata_dir = os.path.normpath(tessdata_dir)
            
            # Set environment variable
            os.environ["TESSDATA_PREFIX"] = tessdata_dir
            print(f"✅ Set TESSDATA_PREFIX to: {tessdata_dir}")
            
            # 3. Ensure language data files exist
            ensure_language_data(tessdata_dir)
            
            return True
        else:
            print("❌ Could not find tessdata directory")
            return False
            
    except ImportError:
        print("❌ pytesseract module not installed")
        return False
    except Exception as e:
        print(f"❌ Error configuring Tesseract: {e}")
        return False

def find_tesseract_executable():
    """Find Tesseract executable path"""
    common_paths = []
    
    # Windows common paths
    if platform.system() == "Windows":
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.join(os.environ.get('LOCALAPPDATA', ''), "Tesseract-OCR", "tesseract.exe"),
            os.path.join(os.environ.get('PROGRAMFILES', ''), "Tesseract-OCR", "tesseract.exe"),
            os.path.join(os.environ.get('PROGRAMFILES(X86)', ''), "Tesseract-OCR", "tesseract.exe")
        ]
    
    # macOS common paths
    elif platform.system() == "Darwin":
        common_paths = [
            "/usr/local/bin/tesseract",
            "/opt/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract"
        ]
    
    # Linux common paths
    else:
        common_paths = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/opt/tesseract/bin/tesseract"
        ]
    
    # Check if any of these paths exist
    for path in common_paths:
        if os.path.isfile(path):
            return path
            
    # Try using the 'where' or 'which' command
    try:
        if platform.system() == "Windows":
            from subprocess import PIPE, run
            result = run(["where", "tesseract"], stdout=PIPE, stderr=PIPE, universal_newlines=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
        else:
            from subprocess import PIPE, run
            result = run(["which", "tesseract"], stdout=PIPE, stderr=PIPE, universal_newlines=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
    except:
        pass
        
    return None

def find_tessdata_dir(tesseract_cmd):
    """Find the tessdata directory based on the tesseract executable path"""
    # Check if TESSDATA_PREFIX is already set
    tessdata_env = os.environ.get("TESSDATA_PREFIX")
    if tessdata_env and os.path.isdir(tessdata_env):
        return tessdata_env
    
    if tesseract_cmd and os.path.exists(tesseract_cmd):
        # Get parent directory of executable
        bin_dir = os.path.dirname(tesseract_cmd)
        parent_dir = os.path.dirname(bin_dir)
        
        # Check common relative paths
        possible_tessdata = [
            os.path.join(parent_dir, "tessdata"),
            os.path.join(parent_dir, "share", "tesseract-ocr", "tessdata"),
            os.path.join(parent_dir, "share", "tessdata"),
            os.path.join(bin_dir, "tessdata"),
        ]
        
        for path in possible_tessdata:
            if os.path.isdir(path):
                return path
    
    return None

def ensure_language_data(tessdata_dir):
    """Ensure language data files for 'eng' and 'en' exist"""
    if not tessdata_dir or not os.path.isdir(tessdata_dir):
        return False
    
    # Check for 'eng.traineddata'
    eng_file = os.path.join(tessdata_dir, "eng.traineddata")
    eng_exists = os.path.isfile(eng_file)
    
    # Check for 'en.traineddata'
    en_file = os.path.join(tessdata_dir, "en.traineddata")
    en_exists = os.path.isfile(en_file)
    
    if eng_exists and not en_exists:
        # Create a copy of eng.traineddata as en.traineddata
        try:
            shutil.copy2(eng_file, en_file)
            print(f"✅ Created en.traineddata from eng.traineddata in {tessdata_dir}")
            return True
        except Exception as e:
            print(f"❌ Failed to create en.traineddata: {e}")
            return False
    elif not eng_exists and en_exists:
        # Create a copy of en.traineddata as eng.traineddata
        try:
            shutil.copy2(en_file, eng_file)
            print(f"✅ Created eng.traineddata from en.traineddata in {tessdata_dir}")
            return True
        except Exception as e:
            print(f"❌ Failed to create eng.traineddata: {e}")
            return False
    elif not eng_exists and not en_exists:
        print(f"❌ No language data files found in {tessdata_dir}")
        return False
        
    return True