"""
EyeShot AI - OCR Result Cache
Caches OCR results to improve performance for repeated operations
Last updated: 2025-06-20 10:46:08 UTC
Author: Tigran0000
"""

import os
import json
import hashlib
import time
import io
from typing import Dict, Any, Optional, Union
from PIL import Image

class ResultCache:
    """
    Caches OCR results to avoid reprocessing the same images.
    Provides both in-memory and disk-based caching options.
    """
    
    def __init__(self, cache_dir: str = None, memory_size: int = 100, max_size: int = None, enable_disk_cache: bool = True):
        """
        Initialize the cache
        
        Args:
            cache_dir: Directory to store cached results
            memory_size: Number of results to keep in memory (deprecated)
            max_size: Number of results to keep in memory
            enable_disk_cache: Whether to use disk-based caching
        """
        # Handle parameter compatibility
        self.memory_size = max_size if max_size is not None else memory_size
            
        # Cache settings
        self.enable_disk_cache = enable_disk_cache
        
        # In-memory cache (most recent results)
        self.memory_cache = {}
        self.access_order = []  # Track access order for LRU eviction
        
        # Disk cache settings
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser('~'), '.eyeshot_cache')
        if self.enable_disk_cache and not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir)
            except Exception as e:
                print(f"⚠️ Failed to create cache directory: {e}")
                self.enable_disk_cache = False
    
    # Interface method to match OCREngine expectations
    def generate_key(self, image, mode="auto", preprocess=True):
        """
        Generate cache key to match OCREngine interface
        
        Args:
            image: PIL Image to generate key for
            mode: Extraction mode
            preprocess: Whether preprocessing was applied
            
        Returns:
            Cache key as string
        """
        try:
            # Create a small thumbnail for hashing to reduce memory usage
            thumb = image.copy()
            thumb.thumbnail((100, 100))
            
            # Convert to grayscale for more stable hashing
            thumb = thumb.convert('L')
            
            # Get image bytes
            img_bytes = io.BytesIO()
            thumb.save(img_bytes, format='PNG')
            
            # Create hash
            hash_obj = hashlib.md5(img_bytes.getvalue())
            
            # Add mode and preprocess to the hash
            hash_obj.update(f"{mode}_{preprocess}".encode('utf-8'))
            
            return hash_obj.hexdigest()
        except Exception as e:
            print(f"Error generating cache key: {e}")
            # Generate a timestamp-based key if normal key generation fails
            return f"fallback_{int(time.time())}"

    # Interface method to match OCREngine expectations
    def get(self, key):
        """
        Get cached result by key
        
        Args:
            key: Cache key to lookup
            
        Returns:
            Cached result if found, None otherwise
        """
        try:
            # Check memory cache first
            if key in self.memory_cache:
                # Update access order (move to end)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.memory_cache[key]
            
            # Check disk cache if enabled
            if self.enable_disk_cache:
                cache_file = os.path.join(self.cache_dir, f"{key}.json")
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                            
                        # Add to memory cache for faster access next time
                        self._add_to_memory_cache(key, result)
                        return result
                    except Exception as e:
                        print(f"⚠️ Cache read error: {e}")
            
            return None
        except Exception as e:
            print(f"Cache access error: {e}")
            return None
    
    # Interface method to match OCREngine expectations
    def set(self, key, value):
        """
        Store result in cache
        
        Args:
            key: Cache key
            value: Result to cache
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Validate inputs
            if not key or not value:
                return False
                
            # Only cache successful results
            if isinstance(value, dict) and not value.get('success', False):
                return False
                
            # Add to memory cache
            self._add_to_memory_cache(key, value)
            
            # Add to disk cache if enabled
            if self.enable_disk_cache:
                cache_file = os.path.join(self.cache_dir, f"{key}.json")
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(value, f)
                    return True
                except Exception as e:
                    print(f"⚠️ Cache write error: {e}")
                    return False
            
            return True
        except Exception as e:
            print(f"Cache store error: {e}")
            return False
    
    # Interface method to match OCREngine expectations
    def clear(self):
        """
        Clear all cached results
        
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            # Clear memory cache
            self.memory_cache = {}
            self.access_order = []
            
            # Clear disk cache if enabled
            if self.enable_disk_cache:
                try:
                    for filename in os.listdir(self.cache_dir):
                        if filename.endswith('.json'):
                            os.remove(os.path.join(self.cache_dir, filename))
                except Exception as e:
                    print(f"⚠️ Cache clear error: {e}")
                    return False
            
            return True
        except Exception as e:
            print(f"Cache clear error: {e}")
            return False
    
    def _add_to_memory_cache(self, key: str, result: Dict[str, Any]):
        """Add result to memory cache with LRU eviction policy"""
        # Check if key already exists
        if key in self.memory_cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            
            # Update value
            self.memory_cache[key] = result
            return
            
        # Check if cache is full
        if len(self.memory_cache) >= self.memory_size:
            # Evict least recently used item
            lru_key = self.access_order.pop(0)
            del self.memory_cache[lru_key]
        
        # Add new item
        self.memory_cache[key] = result
        self.access_order.append(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        memory_items = len(self.memory_cache)
        
        disk_items = 0
        disk_size = 0
        if self.enable_disk_cache and os.path.exists(self.cache_dir):
            try:
                files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
                disk_items = len(files)
                disk_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in files)
            except:
                pass
                
        return {
            'memory_items': memory_items,
            'memory_capacity': self.memory_size,
            'memory_usage_percent': (memory_items / max(1, self.memory_size)) * 100,
            'disk_items': disk_items,
            'disk_size_kb': disk_size / 1024,
            'disk_enabled': self.enable_disk_cache
        }