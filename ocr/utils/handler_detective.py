"""
EyeShot AI - Handler Detection Tool
Identifies which OCR handler processed specific text
Author: Tigran0000
Date: 2025-06-20 15:26:36 UTC
"""

import re
from typing import Dict, List
from datetime import datetime

class HandlerDetective:
    """Analyzes extracted text to determine which handler processed it"""
    
    def __init__(self):
        self.user = "Tigran0000"
        self.timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        # Handler signatures based on your existing handlers
        self.handler_signatures = {
            'book_handler': {
                'patterns': [
                    r'Rule \d+\.',           # Section numbering like "Rule 2."
                    r'IN \d{4},',            # Year patterns like "IN 1958,"
                    r'Chapter \d+',          # Chapter numbering
                    r'[A-Z]{2,}\s+[A-Z][a-z]+',  # Proper name formatting
                ],
                'content_indicators': [
                    'leaders', 'team', 'business', 'company', 'management',
                    'leadership', 'organization', 'corporate', 'strategy'
                ],
                'quality_checks': [
                    ('clean_punctuation', lambda t: len(re.findall(r'\s[,.;:]', t)) < 3),
                    ('proper_spacing', lambda t: len(re.findall(r'[a-z][A-Z]', t)) < 3),
                    ('good_paragraphs', lambda t: '\n\n' in t),
                ]
            },
            'academic_handler': {
                'patterns': [
                    r'\bet al\.',
                    r'\bibid\.',
                    r'bibliography',
                    r'[A-Z][a-z]+, [A-Z]\.',  # Academic citation format
                    r'\d{4}[a-z]?\)',         # Year citations
                ],
                'content_indicators': [
                    'bibliography', 'academic', 'journal', 'analysis', 'research',
                    'study', 'scholar', 'university', 'publication', 'manuscript'
                ],
                'quality_checks': [
                    ('scholarly_terms', lambda t: any(w in t.lower() for w in ['bibliography', 'academic', 'journal'])),
                    ('proper_citations', lambda t: '"' in t and '.' in t),
                ]
            },
            'receipt_handler': {
                'patterns': [
                    r'\$\d+\.\d{2}',         # Currency amounts
                    r'TOTAL',
                    r'TAX',
                    r'\d{2}/\d{2}/\d{4}',    # Date patterns
                ],
                'content_indicators': [
                    'total', 'tax', 'receipt', 'purchase', 'sale', 'payment'
                ],
                'quality_checks': [
                    ('has_prices', lambda t: '$' in t),
                    ('receipt_terms', lambda t: any(w in t.upper() for w in ['TOTAL', 'TAX', 'RECEIPT']))
                ]
            }
        }
    
    def analyze_extraction(self, text: str) -> Dict:
        """Analyze text to determine which handler likely processed it"""
        
        results = {
            'text_sample': text[:200] + "..." if len(text) > 200 else text,
            'analysis_timestamp': self.timestamp,
            'user': self.user,
            'handler_scores': {},
            'most_likely_handler': None,
            'confidence': 0,
            'quality_metrics': {},
            'evidence': [],
            'recommendation': ""
        }
        
        # Score each handler
        for handler_name, signature in self.handler_signatures.items():
            score = self._calculate_handler_score(text, signature)
            results['handler_scores'][handler_name] = score
        
        # Find best match
        if results['handler_scores']:
            best_handler = max(results['handler_scores'], key=results['handler_scores'].get)
            results['most_likely_handler'] = best_handler
            results['confidence'] = results['handler_scores'][best_handler]
        
        # Quality assessment
        results['quality_metrics'] = self._assess_quality(text)
        results['evidence'] = self._gather_evidence(text, results['most_likely_handler'])
        results['recommendation'] = self._make_recommendation(results)
        
        return results
    
    def _calculate_handler_score(self, text: str, signature: Dict) -> float:
        """Calculate score for a specific handler"""
        score = 0
        
        # Pattern matching (30 points max)
        pattern_score = 0
        for pattern in signature['patterns']:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            pattern_score += min(matches * 10, 15)  # 10 points per match, max 15 per pattern
        score += min(pattern_score, 30)
        
        # Content indicators (40 points max)
        content_score = 0
        text_lower = text.lower()
        for indicator in signature['content_indicators']:
            if indicator in text_lower:
                content_score += 8
        score += min(content_score, 40)
        
        # Quality checks (30 points max)
        quality_score = 0
        for check_name, check_func in signature['quality_checks']:
            try:
                if check_func(text):
                    quality_score += 10
            except:
                pass
        score += min(quality_score, 30)
        
        return score
    
    def _assess_quality(self, text: str) -> Dict:
        """Assess overall text extraction quality"""
        
        # Count potential OCR errors
        ocr_errors = 0
        ocr_errors += len(re.findall(r'[a-z][A-Z]', text))      # Joined words
        ocr_errors += len(re.findall(r'\s[,.;:]', text))        # Space before punctuation
        ocr_errors += len(re.findall(r'[,.;:][a-zA-Z]', text))  # Missing space after punctuation
        ocr_errors += len(re.findall(r'[Il1]{2,}', text))       # I/l/1 confusion
        
        # Calculate quality score
        total_chars = len(text)
        error_rate = (ocr_errors / max(total_chars, 1)) * 100
        
        quality_score = max(0, 100 - (error_rate * 50))  # Penalize errors heavily
        
        return {
            'ocr_errors': ocr_errors,
            'error_rate_percent': round(error_rate, 2),
            'quality_score': round(quality_score, 1),
            'quality_level': 'excellent' if quality_score > 90 else 'good' if quality_score > 70 else 'fair' if quality_score > 50 else 'poor'
        }
    
    def _gather_evidence(self, text: str, handler: str) -> List[str]:
        """Gather evidence for the handler identification"""
        evidence = []
        
        # General quality evidence
        if len(re.findall(r'[a-z][A-Z]', text)) < 3:
            evidence.append("✅ Clean word boundaries (minimal joined words)")
        
        if len(re.findall(r'\s[,.;:]', text)) < 2:
            evidence.append("✅ Proper punctuation spacing")
        
        if '\n\n' in text:
            evidence.append("✅ Preserved paragraph structure")
        
        # Handler-specific evidence
        if handler == 'book_handler':
            if re.search(r'Rule \d+\.', text):
                evidence.append("📚 Book section formatting detected")
            if any(term in text.lower() for term in ['leaders', 'chapter', 'business']):
                evidence.append("📚 Business/leadership book content")
            evidence.append("📚 High-quality book handler extraction")
        
        elif handler == 'academic_handler':
            if any(term in text.lower() for term in ['bibliography', 'academic', 'journal']):
                evidence.append("🎓 Academic content detected")
            evidence.append("🎓 Scholarly document processing")
        
        return evidence
    
    def _make_recommendation(self, results: Dict) -> str:
        """Make recommendation based on analysis"""
        
        confidence = results['confidence']
        quality = results['quality_metrics']['quality_score']
        handler = results['most_likely_handler']
        
        if confidence > 80 and quality > 85:
            return f"🎯 EXCELLENT: {handler} is working perfectly for this document type!"
        elif confidence > 60 and quality > 70:
            return f"✅ GOOD: {handler} is handling this document well."
        elif confidence > 40:
            return f"⚠️ FAIR: {handler} may need optimization for this content."
        else:
            return "❌ POOR: Consider trying a different handler or improving preprocessing."