# src/ocr/confidence.py
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import re
from collections import Counter

class ConfidenceScorer:
    """Confidence scoring system for OCR engine results"""
    
    # Confidence thresholds
    LOW_CONFIDENCE = 40
    MEDIUM_CONFIDENCE = 70
    HIGH_CONFIDENCE = 90
    
    # Weight factors for different aspects of confidence
    WEIGHTS = {
        'ocr_confidence': 0.6,      # Raw OCR engine confidence
        'text_quality': 0.2,        # Text quality metrics
        'structure_quality': 0.1,   # Structure preservation quality
        'engine_reliability': 0.1   # Engine reliability for document type
    }
    
    # Engine reliability ratings for different document types
    ENGINE_RELIABILITY = {
        'tesseract': {
            'standard': 0.9,
            'academic': 0.8,
            'handwritten': 0.5,
            'receipt': 0.7,
            'code': 0.8,
            'table': 0.7,
            'form': 0.8,
            'id_card': 0.7,
            'title': 0.8,
            'math': 0.6
        },
        'easyocr': {
            'standard': 0.8,
            'academic': 0.7,
            'handwritten': 0.9,
            'receipt': 0.6,
            'code': 0.6,
            'table': 0.6,
            'form': 0.7,
            'id_card': 0.8,
            'title': 0.9,
            'math': 0.5
        },
        'paddleocr': {
            'standard': 0.8,
            'academic': 0.7,
            'handwritten': 0.8,
            'receipt': 0.7,
            'code': 0.5,
            'table': 0.6,
            'form': 0.7,
            'id_card': 0.9,
            'title': 0.9,
            'math': 0.5
        }
    }
    
    def __init__(self):
        """Initialize confidence scorer with default settings"""
        # Language model for text quality assessment (loaded on demand)
        self.language_model = None
        self.enable_language_checking = False
    
    def calculate_confidence(self, 
                            raw_confidence: float, 
                            text: str, 
                            engine: str, 
                            document_type: str, 
                            structure_preserved: bool = False) -> float:
        """
        Calculate overall confidence score based on multiple factors
        
        Args:
            raw_confidence: Raw OCR engine confidence score (0-100)
            text: Extracted text content
            engine: OCR engine used ('tesseract', 'easyocr', 'paddleocr')
            document_type: Type of document processed
            structure_preserved: Whether structure was preserved
            
        Returns:
            Overall confidence score (0-100)
        """
        # Normalize raw confidence to 0-100 scale
        ocr_confidence = self._normalize_raw_confidence(raw_confidence, engine)
        
        # Calculate text quality score
        text_quality = self._calculate_text_quality(text, document_type)
        
        # Calculate structure quality
        structure_quality = 100 if structure_preserved else 50
        
        # Get engine reliability for document type
        engine_reliability = self._get_engine_reliability(engine, document_type) * 100
        
        # Calculate weighted score
        weighted_score = (
            ocr_confidence * self.WEIGHTS['ocr_confidence'] +
            text_quality * self.WEIGHTS['text_quality'] +
            structure_quality * self.WEIGHTS['structure_quality'] +
            engine_reliability * self.WEIGHTS['engine_reliability']
        )
        
        return round(weighted_score, 1)
    
    def get_confidence_level(self, score: float) -> str:
        """Convert numerical confidence score to descriptive level"""
        if score >= self.HIGH_CONFIDENCE:
            return "high"
        elif score >= self.MEDIUM_CONFIDENCE:
            return "medium"
        elif score >= self.LOW_CONFIDENCE:
            return "low"
        else:
            return "very low"
    
    def filter_low_confidence_words(self, 
                                  data: Dict[str, Any], 
                                  threshold: float = 30) -> Dict[str, Any]:
        """
        Filter out words below confidence threshold from OCR data
        
        Args:
            data: OCR data dictionary with 'text', 'conf', etc. arrays
            threshold: Confidence threshold (0-100)
            
        Returns:
            Filtered OCR data dictionary
        """
        if 'text' not in data or 'conf' not in data:
            return data
        
        # Create mask for confidence values above threshold
        mask = [int(conf) > threshold for conf in data['conf']]
        
        # Apply mask to all arrays in data
        filtered_data = {}
        for key, values in data.items():
            filtered_data[key] = [v for v, m in zip(values, mask) if m]
        
        return filtered_data
    
    def adjust_confidence_for_mode(self, confidence: float, mode: str) -> float:
        """Adjust confidence score based on extraction mode"""
        # Different modes have different baseline confidence levels
        mode_factors = {
            'standard': 1.0,
            'academic': 0.95,
            'title': 1.1,        # Titles are usually clearer
            'handwritten': 0.8,  # Handwritten text is less reliable
            'receipt': 0.9,
            'code': 0.95,
            'table': 0.9,
            'form': 0.95,
            'id_card': 1.0,
            'math': 0.8,         # Math formulas are difficult
            'mixed': 0.85        # Mixed content is challenging
        }
        
        # Apply mode-specific factor
        factor = mode_factors.get(mode, 1.0)
        adjusted = confidence * factor
        
        # Ensure confidence stays in valid range
        return max(0, min(adjusted, 100))
    
    def analyze_confidence_distribution(self, confidences: List[float]) -> Dict[str, Any]:
        """
        Analyze distribution of confidence scores
        
        Args:
            confidences: List of confidence values
            
        Returns:
            Dictionary with statistical analysis
        """
        if not confidences:
            return {
                'min': 0,
                'max': 0,
                'mean': 0,
                'median': 0,
                'std_dev': 0,
                'low_confidence_percent': 0,
                'high_confidence_percent': 0,
                'histogram': {}
            }
        
        confidences = np.array(confidences)
        
        # Calculate histogram (binned distribution)
        hist, bin_edges = np.histogram(confidences, bins=10, range=(0, 100))
        histogram = {f"{int(bin_edges[i])}-{int(bin_edges[i+1])}": int(hist[i]) 
                    for i in range(len(hist))}
        
        # Calculate statistics
        return {
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'mean': float(np.mean(confidences)),
            'median': float(np.median(confidences)),
            'std_dev': float(np.std(confidences)),
            'low_confidence_percent': float(np.mean(confidences < self.LOW_CONFIDENCE) * 100),
            'high_confidence_percent': float(np.mean(confidences >= self.HIGH_CONFIDENCE) * 100),
            'histogram': histogram
        }
    
    def combine_engine_confidences(self, 
                                 confidences: Dict[str, float],
                                 document_type: str) -> float:
        """
        Combine confidence scores from multiple OCR engines
        
        Args:
            confidences: Dict mapping engine names to their confidence scores
            document_type: Type of document being processed
            
        Returns:
            Combined confidence score
        """
        if not confidences:
            return 0.0
            
        # Weight each engine based on reliability for document type
        engine_weights = {}
        weight_sum = 0
        
        for engine, confidence in confidences.items():
            weight = self._get_engine_reliability(engine, document_type)
            engine_weights[engine] = weight
            weight_sum += weight
        
        # Normalize weights
        if weight_sum > 0:
            normalized_weights = {e: w/weight_sum for e, w in engine_weights.items()}
        else:
            # Equal weights if no reliability data
            normalized_weights = {e: 1.0/len(confidences) for e in confidences}
        
        # Calculate weighted average
        weighted_sum = sum(confidences[e] * normalized_weights[e] for e in confidences)
        
        return weighted_sum
    
    def estimate_confidence_from_text(self, text: str, document_type: str) -> float:
        """
        Estimate confidence based on text characteristics when no score is available
        
        Args:
            text: Extracted text
            document_type: Type of document
            
        Returns:
            Estimated confidence score (0-100)
        """
        if not text:
            return 0.0
        
        # Calculate text quality metrics
        metrics = self._get_text_quality_metrics(text)
        
        # Base confidence based on coherence metrics
        base_confidence = 70.0  # Start with moderate confidence
        
        # Adjust based on metrics
        adjustments = [
            # More words generally means better extraction
            min(metrics['word_count'] / 5, 10),
            
            # Penalize nonsense character sequences
            -min(metrics['nonsense_words'] * 5, 20),
            
            # High ratio of valid dictionary words is good
            min(metrics['dictionary_word_ratio'] * 10, 10),
            
            # Penalize abnormal punctuation
            -min(metrics['abnormal_punct_ratio'] * 15, 15)
        ]
        
        # Apply adjustments
        adjusted_confidence = base_confidence + sum(adjustments)
        
        # Ensure confidence is in valid range
        return max(0, min(adjusted_confidence, 100))
    
    def _normalize_raw_confidence(self, confidence: float, engine: str) -> float:
        """Normalize confidence score to 0-100 scale based on engine"""
        if confidence < 0:
            return 0
        
        # Some engines use 0-1 scale
        if engine == 'easyocr' and confidence <= 1.0:
            return confidence * 100
        
        # Some engines might have different distributions
        if engine == 'tesseract':
            # Tesseract scores tend to cluster - expand the range
            if confidence > 90:
                # High confidence range expansion
                return 85 + (confidence - 90) * 1.5
            elif confidence < 50:
                # Low confidence range expansion
                return confidence * 0.8
        
        # Ensure confidence is in valid range
        return max(0, min(confidence, 100))
    
    def _calculate_text_quality(self, text: str, document_type: str) -> float:
        """Calculate text quality score based on content analysis"""
        if not text:
            return 0.0
            
        # Get text quality metrics
        metrics = self._get_text_quality_metrics(text)
        
        # Base quality score
        base_quality = 70.0
        
        # Document type specific adjustments
        doc_type_adjustments = {
            'standard': 0,
            'academic': 5 if metrics['avg_word_length'] > 5.0 else -5,
            'handwritten': -10,  # Handwritten text is generally less reliable
            'receipt': 5 if metrics['digit_ratio'] > 0.2 else -5,  # Receipts have many numbers
            'code': 5 if metrics['special_char_ratio'] > 0.1 else -5,  # Code has special chars
            'table': 0,
            'form': 5 if ':' in text else -5,  # Forms typically have field markers
            'id_card': 5 if metrics['alphanum_ratio'] > 0.9 else -5,
            'title': 5 if metrics['avg_word_length'] < 8.0 else -5,  # Titles have shorter words
            'math': 5 if metrics['special_char_ratio'] > 0.2 else -5  # Math has many symbols
        }
        
        # Apply document-specific adjustment
        quality = base_quality + doc_type_adjustments.get(document_type, 0)
        
        # Universal adjustments
        adjustments = [
            # More words generally means better extraction (up to a point)
            min(metrics['word_count'] / 10, 10),
            
            # Penalize nonsense character sequences
            -min(metrics['nonsense_words'] * 5, 20),
            
            # Bonus for dictionary words
            min(metrics['dictionary_word_ratio'] * 10, 15),
            
            # Penalize abnormal punctuation
            -min(metrics['abnormal_punct_ratio'] * 15, 15)
        ]
        
        # Apply adjustments
        quality += sum(adjustments)
        
        # Ensure score is in valid range
        return max(0, min(quality, 100))
    
    def _get_text_quality_metrics(self, text: str) -> Dict[str, float]:
        """Calculate various text quality metrics for confidence estimation"""
        if not text:
            return {
                'word_count': 0,
                'avg_word_length': 0,
                'dictionary_word_ratio': 0,
                'nonsense_words': 0,
                'abnormal_punct_ratio': 0,
                'digit_ratio': 0,
                'special_char_ratio': 0,
                'alphanum_ratio': 0
            }
        
        # Split into words
        words = re.findall(r'\b[A-Za-z]+\b', text)
        all_words = text.split()
        
        # Count characters by type
        char_counts = Counter(text)
        total_chars = len(text)
        
        # Calculate metrics
        metrics = {
            'word_count': len(all_words),
            'avg_word_length': sum(len(w) for w in words) / max(len(words), 1),
            'nonsense_words': self._count_nonsense_words(words),
            'abnormal_punct_ratio': self._calculate_abnormal_punct_ratio(text),
            'digit_ratio': sum(c.isdigit() for c in text) / max(total_chars, 1),
            'special_char_ratio': sum(not c.isalnum() and not c.isspace() for c in text) / max(total_chars, 1),
            'alphanum_ratio': sum(c.isalnum() for c in text) / max(total_chars, 1)
        }
        
        # Dictionary word ratio (basic approximation)
        metrics['dictionary_word_ratio'] = self._estimate_dictionary_word_ratio(words)
        
        return metrics
    
    def _count_nonsense_words(self, words: List[str]) -> int:
        """Count likely nonsense words (abnormal character patterns)"""
        nonsense_count = 0
        
        for word in words:
            # Very short words are OK
            if len(word) <= 2:
                continue
                
            # Check for abnormal character patterns
            if re.search(r'[aeiou]{4}|[bcdfghjklmnpqrstvwxyz]{5}', word.lower()):
                # Too many consecutive vowels or consonants
                nonsense_count += 1
                continue
                
            # Check for unlikely character combinations
            if re.search(r'[jqxz]{2}', word.lower()):
                # Uncommon letters appearing consecutively
                nonsense_count += 1
                continue
                
            # Check vowel presence in longer words
            if len(word) > 3 and not re.search(r'[aeiou]', word.lower()):
                # Longer word with no vowels
                nonsense_count += 1
        
        return nonsense_count
    
    def _calculate_abnormal_punct_ratio(self, text: str) -> float:
        """Calculate ratio of abnormal punctuation patterns"""
        if not text:
            return 0.0
            
        abnormal_patterns = [
            r'[.,:;!?]{2,}',           # Multiple punctuation marks in a row
            r'[a-zA-Z][.,:;!?][a-zA-Z]',  # Punctuation between letters with no space
            r'\s[.,:;!?]\s'            # Isolated punctuation
        ]
        
        abnormal_count = sum(len(re.findall(pattern, text)) for pattern in abnormal_patterns)
        total_punct = sum(c in '.,:;!?()[]{}' for c in text)
        
        return abnormal_count / max(total_punct, 1)
    
    def _estimate_dictionary_word_ratio(self, words: List[str]) -> float:
        """Estimate ratio of words that are likely dictionary words"""
        if not words:
            return 0.0
        
        # For simplicity, use basic heuristics instead of a full dictionary
        likely_real_words = 0
        
        for word in words:
            word = word.lower()
            
            # Words that are too short or too long
            if len(word) <= 1 or len(word) > 25:
                continue
                
            # Check for vowels in normal words
            if len(word) > 3 and not re.search(r'[aeiou]', word):
                continue
                
            # Check for balanced consonant/vowel ratio
            vowel_count = sum(c in 'aeiou' for c in word)
            consonant_count = len(word) - vowel_count
            
            if len(word) > 4 and (vowel_count == 0 or consonant_count == 0):
                continue
                
            # Passed basic checks, likely a real word
            likely_real_words += 1
        
        return likely_real_words / max(len(words), 1)
    
    def _get_engine_reliability(self, engine: str, document_type: str) -> float:
        """Get reliability rating for a specific engine and document type"""
        # Normalize engine name
        engine = engine.lower()
        if 'tesseract' in engine:
            engine = 'tesseract'
        elif 'easyocr' in engine:
            engine = 'easyocr'
        elif 'paddle' in engine:
            engine = 'paddleocr'
        
        # Normalize document type
        document_type = document_type.lower() if document_type else 'standard'
        if document_type not in self.ENGINE_RELIABILITY['tesseract']:
            document_type = 'standard'
        
        # Return reliability rating
        if engine in self.ENGINE_RELIABILITY and document_type in self.ENGINE_RELIABILITY[engine]:
            return self.ENGINE_RELIABILITY[engine][document_type]
        
        # Default reliability for unknown combinations
        return 0.7