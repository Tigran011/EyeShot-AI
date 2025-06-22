import numpy as np
import cv2
import math
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

class ImageProcessor:
    """Enhanced image preprocessing for better OCR results"""
    
    def preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """Standard preprocessing pipeline"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply standard preprocessing steps
        image = self._enhance_contrast(image)
        image = self._convert_to_grayscale(image)
        image = self._apply_threshold(image)
        image = self._denoise(image)
        
        return image
    
    def preprocess_inverted(self, image: Image.Image) -> Image.Image:
        """Preprocessing with color inversion (for dark backgrounds)"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Invert colors first (white text on dark bg becomes black text on white bg)
        image = ImageOps.invert(image)
        
        # Apply standard preprocessing
        image = self._enhance_contrast(image)
        image = self._convert_to_grayscale(image)
        image = self._apply_threshold(image)
        image = self._denoise(image)
        
        return image
    
    def preprocess_high_contrast(self, image: Image.Image) -> Image.Image:
        """High contrast preprocessing"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Aggressive contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.5)  # Very high contrast
        
        # Convert to grayscale
        image = self._convert_to_grayscale(image)
        
        # Apply binary threshold
        image = self._apply_binary_threshold(image)
        
        return image
    
    def preprocess_dark_background(self, image: Image.Image) -> Image.Image:
        """Specialized preprocessing for dark backgrounds with light text"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array for advanced processing
        img_array = np.array(image)
        
        # Check if image has dark background (average brightness < 128)
        gray_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray_array)
        
        if avg_brightness < 128:
            # Dark background detected - invert
            img_array = 255 - img_array
        
        # Convert back to PIL
        image = Image.fromarray(img_array)
        
        # Apply enhanced processing
        image = self._enhance_contrast(image, factor=2.0)
        image = self._convert_to_grayscale(image)
        image = self._apply_threshold(image)
        
        return image
    
    def preprocess_enhanced_edges(self, image: Image.Image) -> Image.Image:
        """Edge enhancement preprocessing"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Convert to grayscale
        image = self._convert_to_grayscale(image)
        
        # Apply edge enhancement
        image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # Apply threshold
        image = self._apply_threshold(image)
        
        return image
    
    def preprocess_stylized_title(self, image: Image.Image) -> Image.Image:
        """Specialized preprocessing for stylized light text on dark backgrounds"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 1. Aggressive color inversion - ensures white text on black becomes black on white
        image = ImageOps.invert(image)
        
        # 2. Convert to numpy for advanced processing
        img_array = np.array(image)
        
        # 3. Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        # 4. Convert to grayscale
        gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
        
        # 5. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 6. Use Otsu's thresholding to find optimal binary threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 7. Morphological operations to sharpen character edges
        kernel = np.ones((2,2), np.uint8)
        processed = cv2.dilate(binary, kernel, iterations=1)
        
        # Return as PIL Image
        return Image.fromarray(processed)

    def preprocess_book_page(self, image: Image.Image) -> Image.Image:
        """
        Specialized preprocessing for book pages with optimal text clarity
        """
        # Convert PIL image to OpenCV format
        cv_img = np.array(image)
        
        # Convert to grayscale if needed
        if len(cv_img.shape) == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_img
        
        # Apply mild Gaussian blur to remove noise while preserving text
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding to binarize the image while handling varying lighting
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 9
        )
        
        # Enhance contrast to make text clearer
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(binary)
        
        # Reduce noise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Convert back to PIL
        return Image.fromarray(denoised)

    def preprocess_academic_text(self, image: Image.Image) -> Image.Image:
        """Specialized preprocessing for academic printed text with decorative elements"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a numpy array for advanced processing
        img_array = np.array(image)
        
        # Check if image has very light background (typical for book scans)
        gray_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray_array)
        
        # For book scans (typically light backgrounds)
        if avg_brightness > 200:
            # 1. Apply light denoising to reduce scanner artifacts while preserving text edges
            denoised = cv2.fastNlMeansDenoising(gray_array, h=10)
            
            # 2. Enhance contrast to make text more distinct
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # 3. Apply gentle threshold to separate text from background
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 4. Apply slight morphological operations to connect broken characters
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return Image.fromarray(processed)
        
        # Default processing for other cases
        return self._convert_to_grayscale(image)
    
    def preprocess_handwritten(self, image: Image.Image) -> Image.Image:
        """Specialized preprocessing for handwritten text"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a numpy array for advanced processing
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # Use adaptive thresholding which works better for handwriting
        binary = cv2.adaptiveThreshold(
            enhanced, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Invert back to black text on white background
        binary = 255 - binary
        
        # Return as PIL Image
        return Image.fromarray(binary)
    
    def preprocess_receipt(self, image: Image.Image) -> Image.Image:
        """Specialized preprocessing for receipt text"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a numpy array for advanced processing
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Deskew the receipt (fix alignment) - receipts are often skewed
        skew_angle = self._get_skew_angle(gray)
        if abs(skew_angle) > 0.5:  # Only correct if skew is significant
            rotated = self._rotate_image(gray, skew_angle)
        else:
            rotated = gray
        
        # Apply noise removal
        denoised = cv2.fastNlMeansDenoising(rotated, h=10)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Return as PIL Image
        return Image.fromarray(binary)
    
    def preprocess_code(self, image: Image.Image) -> Image.Image:
        """Specialized preprocessing for code text"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a numpy array for advanced processing
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply noise removal - gentle to preserve small characters
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding to handle different lighting conditions
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Return as PIL Image
        return Image.fromarray(binary)
    
    def preprocess_table(self, image: Image.Image) -> Image.Image:
        """Specialized preprocessing for table text"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a numpy array for advanced processing
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply noise removal
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Enhance contrast to make table lines more visible
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Return as PIL Image
        return Image.fromarray(binary)
    
    def preprocess_form(self, image: Image.Image) -> Image.Image:
        """Specialized preprocessing for form text"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a numpy array for advanced processing
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply noise removal
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply thresholding
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Return as PIL Image
        return Image.fromarray(binary)
    
    def preprocess_id_card(self, image: Image.Image) -> Image.Image:
        """Specialized preprocessing for ID card text"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a numpy array for advanced processing
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply bilateral filter to preserve edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Return as PIL Image
        return Image.fromarray(binary)
    
    def preprocess_math(self, image: Image.Image) -> Image.Image:
        """Specialized preprocessing for math text"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a numpy array for advanced processing
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply bilateral filter to preserve edges of math symbols
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # Apply thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Return as PIL Image
        return Image.fromarray(binary)
    
    def _enhance_contrast(self, image: Image.Image, factor: float = 1.5) -> Image.Image:
        """Enhance image contrast"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def _convert_to_grayscale(self, image: Image.Image) -> Image.Image:
        """Convert image to grayscale"""
        return image.convert('L')
    
    def _apply_threshold(self, image: Image.Image) -> Image.Image:
        """Apply adaptive threshold to create clean black/white image"""
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Apply adaptive threshold using OpenCV
        threshold_img = cv2.adaptiveThreshold(
            img_array,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Convert back to PIL Image
        return Image.fromarray(threshold_img)
    
    def _apply_binary_threshold(self, image: Image.Image, threshold: int = 127) -> Image.Image:
        """Apply simple binary threshold"""
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Apply binary threshold
        _, threshold_img = cv2.threshold(img_array, threshold, 255, cv2.THRESH_BINARY)
        
        # Convert back to PIL Image
        return Image.fromarray(threshold_img)
    
    def _denoise(self, image: Image.Image) -> Image.Image:
        """Remove noise from image"""
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply median filter to remove noise
        denoised = cv2.medianBlur(img_array, 3)
        
        # Convert back to PIL Image
        return Image.fromarray(denoised)
    
    def _get_skew_angle(self, gray_image):
        """Get skew angle of an image"""
        
        # Edge detection
        edges = cv2.Canny(gray_image, 150, 200, 3, 5)
        
        # Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=5)
        
        if lines is None:
            return 0.0
        
        # Calculate angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # Avoid division by zero
                angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) < 45:  # Consider only near-horizontal lines
                    angles.append(angle)
        
        if angles:
            return np.median(angles)
        else:
            return 0.0
    
    def _rotate_image(self, image, angle):
        """Rotate an image by the given angle"""
        
        # Get image dimensions
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        
        return rotated