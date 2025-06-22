"""
EyeShot AI - Tesseract Configuration Helper
Helps resolve common Tesseract installation and configuration issues
Last updated: 2025-06-20 11:15:22 UTC
Author: Tigran0000
"""

import os
import sys
import platform
import subprocess
import urllib.request
from pathlib import Path

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
    
    # If not found in common locations, try to find it on PATH
    try:
        # On Windows
        if platform.system() == "Windows":
            result = subprocess.run(["where", "tesseract"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
        # On Unix-like systems
        else:
            result = subprocess.run(["which", "tesseract"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
    except:
        pass
    
    return None

def find_tessdata_dir():
    """Find the tessdata directory"""
    # Check environment variable first
    tessdata_env = os.environ.get("TESSDATA_PREFIX")
    if tessdata_env and os.path.isdir(tessdata_env):
        return tessdata_env
    
    # Check common locations based on executable path
    exe_path = find_tesseract_executable()
    if exe_path:
        # Get parent directory of executable
        bin_dir = os.path.dirname(exe_path)
        parent_dir = os.path.dirname(bin_dir)
        
        # Check common relative paths
        possible_tessdata = [
            os.path.join(parent_dir, "tessdata"),
            os.path.join(bin_dir, "tessdata"),
            os.path.join(parent_dir, "share", "tesseract-ocr", "tessdata"),
            os.path.join(parent_dir, "share", "tessdata"),
        ]
        
        for path in possible_tessdata:
            if os.path.isdir(path) and has_traineddata_file(path, 'eng'):
                return path
    
    return None

def has_traineddata_file(directory, lang='eng'):
    """Check if directory has the specified language traineddata file"""
    if not directory or not os.path.isdir(directory):
        return False
    
    traineddata_file = os.path.join(directory, f"{lang}.traineddata")
    return os.path.isfile(traineddata_file)

def download_language_file(lang="eng"):
    """Download a language data file"""
    GITHUB_BASE_URL = "https://github.com/tesseract-ocr/tessdata/raw/main/"
    
    # Get tessdata directory
    tessdata_dir = find_tessdata_dir()
    if not tessdata_dir:
        # Try creating a tessdata directory in executable location
        exe_path = find_tesseract_executable()
        if exe_path:
            tessdata_dir = os.path.join(os.path.dirname(exe_path), "tessdata")
            os.makedirs(tessdata_dir, exist_ok=True)
        else:
            print("❌ Could not find or create tessdata directory")
            return False
    
    # File paths
    lang_file = f"{lang}.traineddata"
    local_file = os.path.join(tessdata_dir, lang_file)
    download_url = f"{GITHUB_BASE_URL}{lang_file}"
    
    print(f"⬇️ Downloading {lang} language data from GitHub...")
    
    try:
        # Download file
        urllib.request.urlretrieve(download_url, local_file)
        
        if os.path.exists(local_file):
            print(f"✅ Successfully downloaded {lang} language data to {local_file}")
            return True
        else:
            print(f"❌ Failed to download language data")
            return False
    except Exception as e:
        print(f"❌ Error downloading language file: {e}")
        return False

def fix_tesseract_installation():
    """Fix common Tesseract issues"""
    print("🔍 Diagnosing Tesseract installation...")
    
    # Step 1: Find Tesseract executable
    exe_path = find_tesseract_executable()
    if exe_path:
        print(f"✅ Found Tesseract executable at: {exe_path}")
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = exe_path
        except ImportError:
            print("❌ pytesseract module is not installed. Please install it with: pip install pytesseract")
            return False
    else:
        print("❌ Tesseract executable not found. Please install Tesseract OCR.")
        return False
    
    # Step 2: Find tessdata directory with proper normalization of path separators
    tessdata_dir = find_tessdata_dir()
    if tessdata_dir:
        # Normalize path with proper OS-specific path separators
        tessdata_dir = os.path.normpath(tessdata_dir)
        print(f"✅ Found tessdata directory at: {tessdata_dir}")
    else:
        print("❌ tessdata directory not found.")
        
        # Try creating tessdata directory at default location
        default_tessdata = os.path.join(os.path.dirname(exe_path), "tessdata")
        default_tessdata = os.path.normpath(default_tessdata)
        try:
            os.makedirs(default_tessdata, exist_ok=True)
            print(f"✅ Created tessdata directory at: {default_tessdata}")
            tessdata_dir = default_tessdata
        except:
            print("❌ Failed to create tessdata directory")
            return False
    
    # Step 3: Set environment variable with normalized path
    os.environ["TESSDATA_PREFIX"] = tessdata_dir
    print(f"✅ Set TESSDATA_PREFIX environment variable to: {tessdata_dir}")
    
    # Step 4: Check for language data
    eng_file = os.path.join(tessdata_dir, "eng.traineddata")
    if os.path.isfile(eng_file):
        print("✅ English language data found")
    else:
        print("❌ English language data not found. Downloading...")
        if download_language_file("eng"):
            print("✅ Successfully downloaded English language data")
        else:
            print("❌ Failed to download English language data")
            return False
    
    # Step 5: Check for language data - English default
    lang_file = os.path.join(tessdata_dir, "en.traineddata")
    if os.path.isfile(lang_file):
        print("✅ 'en' language data found")
    else:
        print("❌ 'en' language data not found, but 'eng' was found. Creating symlink or copy...")
        try:
            # Copy eng.traineddata to en.traineddata
            import shutil
            shutil.copy2(eng_file, lang_file)
            print(f"✅ Copied eng.traineddata to en.traineddata")
        except Exception as e:
            print(f"❌ Failed to create en.traineddata: {e}")
            return False
    
    # Step 6: Write direct path to file for applications that ignore environment variables
    try:
        # Optional: Create a config file that explicitly sets the path
        import pytesseract
        with open("tessdata_path.txt", "w") as f:
            f.write(tessdata_dir)
        print("✅ Saved tessdata path to config file")
    except Exception:
        # Not critical if this fails
        pass
    
    # Step 7: Verify installation
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        
        # Force setting the path programmatically as well
        pytesseract.pytesseract.tesseract_cmd = exe_path
        
        # Test calling with explicit tessdata path
        test_text = pytesseract.image_to_string(
            Image.new('RGB', (100, 100), color=(255, 255, 255)),
            config=f'--tessdata-dir "{tessdata_dir}"'
        )
        
        print(f"✅ Tesseract is now properly configured (version: {version})")
        return True
    except NameError:
        # Image not imported yet, just verify version
        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract is now properly configured (version: {version})")
            return True
        except Exception as e:
            print(f"❌ Tesseract verification failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Tesseract verification failed: {e}")
        return False

if __name__ == "__main__":
    # When run directly, try to fix the installation
    fix_tesseract_installation()