from typing import Dict, Optional, Union, Any
from functools import lru_cache
from .dictionary import DEFAULT_DICTIONARY

class Translator:
    # 全局默认实例
    _default_instance = None
    
    def __init__(self, to_lang: str = 'en',
                 dictionary: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        self.to_lang = to_lang
        self.dictionary = dictionary or DEFAULT_DICTIONARY
        self._cache = {}  # 简单的缓存机制
    
    @classmethod
    def get_default(cls) -> 'Translator':
        """获取全局默认实例"""
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance
    
    def __call__(self, key: str, lang: Optional[str] = None) -> str:
        return self.translate(key, lang)
    
    def translate(self, key: str, lang: Optional[str] = None) -> str:
        target_lang = lang or self.to_lang
        
        # 检查缓存
        cache_key = f"{key}_{target_lang}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 获取翻译
        translations = self.dictionary.get(key, {})
        result = translations.get(target_lang, key)
        
        # 缓存结果
        self._cache[cache_key] = result
        return result
    
    def add_translation(self, key: str, lang: str, translation: str) -> None:
        if key not in self.dictionary:
            self.dictionary[key] = {}
        self.dictionary[key][lang] = translation
        
        # 清除相关缓存
        self._clear_cache_for_key(key)
    
    def remove_translation(self, key: str, lang: str) -> None:
        if key in self.dictionary:
            if lang in self.dictionary[key]:
                del self.dictionary[key][lang]
                if not self.dictionary[key]:
                    del self.dictionary[key]
                
                # 清除相关缓存
                self._clear_cache_for_key(key)
    
    def _clear_cache_for_key(self, key: str) -> None:
        """清除与特定键相关的所有缓存"""
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{key}_")]
        for k in keys_to_remove:
            del self._cache[k]
    
    def clear_cache(self) -> None:
        """清除所有缓存"""
        self._cache.clear()
    
    def get_dictionary(self) -> Dict[str, Dict[str, str]]:
        return self.dictionary
    
    def set_dictionary(self, dictionary: Dict[str, Dict[str, str]]) -> None:
        self.dictionary = dictionary
        self.clear_cache()  # 字典改变时清除所有缓存

# 使用LRU缓存优化全局函数
@lru_cache(maxsize=1024)
def _cached_translate(key: str, lang: str, dictionary_id: int = 0) -> str:
    """带缓存的翻译函数，dictionary_id用于区分不同的字典"""
    translator = Translator.get_default()
    return translator.translate(key, lang)

def translate(key: str, lang: str = 'en', 
             dictionary: Optional[Union[Dict[str, Dict[str, str]], Translator]] = None) -> str:
    if dictionary is None:
        # 使用默认翻译器并利用缓存
        return _cached_translate(key, lang, id(Translator.get_default().dictionary))
    elif isinstance(dictionary, Translator):
        return dictionary.translate(key, lang)
    elif isinstance(dictionary, dict):
        # 对于临时字典，不使用缓存
        return dictionary.get(key, {}).get(lang, key)
    else:
        raise ValueError("Invalid dictionary type")