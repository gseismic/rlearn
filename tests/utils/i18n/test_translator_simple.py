#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻译器测试文件 - 简化版本
"""

import unittest
import time
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from rlearn.utils.i18n.translator import Translator, translate, _cached_translate


class TestTranslator(unittest.TestCase):
    """翻译器测试类"""
    
    def setUp(self):
        """测试前的设置"""
        # 创建测试字典
        self.test_dictionary = {
            'hello': {
                'zh': '你好',
                'en': 'Hello',
                'ja': 'こんにちは'
            },
            'world': {
                'zh': '世界',
                'en': 'World',
                'ja': '世界'
            },
            'test_key': {
                'zh': '测试键',
                'en': 'Test Key'
            }
        }
        
        # 创建翻译器实例
        self.translator = Translator(to_lang='zh', dictionary=self.test_dictionary)
    
    def test_basic_translation(self):
        """测试基本翻译功能"""
        print("测试基本翻译功能...")
        
        # 测试默认语言翻译
        self.assertEqual(self.translator('hello'), '你好')
        self.assertEqual(self.translator('world'), '世界')
        
        # 测试指定语言翻译
        self.assertEqual(self.translator('hello', 'en'), 'Hello')
        self.assertEqual(self.translator('world', 'ja'), '世界')
        
        # 测试不存在的键（应该返回原键）
        self.assertEqual(self.translator('nonexistent'), 'nonexistent')
        
        # 测试不存在的语言（应该返回原键）
        self.assertEqual(self.translator('hello', 'fr'), 'hello')
        
        print("✅ 基本翻译功能测试通过")
    
    def test_callable_interface(self):
        """测试可调用接口"""
        print("测试可调用接口...")
        
        # 测试实例可以直接调用
        self.assertEqual(self.translator('hello'), '你好')
        self.assertEqual(self.translator('world', 'en'), 'World')
        
        print("✅ 可调用接口测试通过")
    
    def test_add_translation(self):
        """测试添加翻译功能"""
        print("测试添加翻译功能...")
        
        # 添加新翻译
        self.translator.add_translation('new_key', 'zh', '新键')
        self.translator.add_translation('new_key', 'en', 'New Key')
        
        # 验证翻译
        self.assertEqual(self.translator('new_key'), '新键')
        self.assertEqual(self.translator('new_key', 'en'), 'New Key')
        
        # 验证字典已更新
        self.assertIn('new_key', self.translator.dictionary)
        self.assertEqual(self.translator.dictionary['new_key']['zh'], '新键')
        
        print("✅ 添加翻译功能测试通过")
    
    def test_remove_translation(self):
        """测试删除翻译功能"""
        print("测试删除翻译功能...")
        
        # 删除特定语言的翻译
        self.translator.remove_translation('hello', 'ja')
        
        # 验证日语翻译已删除
        self.assertNotIn('ja', self.translator.dictionary['hello'])
        
        # 验证其他语言翻译仍然存在
        self.assertEqual(self.translator('hello'), '你好')
        self.assertEqual(self.translator('hello', 'en'), 'Hello')
        
        # 删除所有翻译
        self.translator.remove_translation('test_key', 'zh')
        self.translator.remove_translation('test_key', 'en')
        
        # 验证整个键被删除
        self.assertNotIn('test_key', self.translator.dictionary)
        
        print("✅ 删除翻译功能测试通过")
    
    def test_cache_functionality(self):
        """测试缓存功能"""
        print("测试缓存功能...")
        
        # 第一次翻译（会缓存）
        result1 = self.translator('hello')
        
        # 第二次翻译（应该从缓存获取）
        result2 = self.translator('hello')
        
        # 结果应该相同
        self.assertEqual(result1, result2)
        
        # 验证缓存存在
        cache_key = 'hello_zh'
        self.assertIn(cache_key, self.translator._cache)
        
        # 测试缓存清除
        self.translator.clear_cache()
        self.assertEqual(len(self.translator._cache), 0)
        
        print("✅ 缓存功能测试通过")
    
    def test_cache_invalidation(self):
        """测试缓存失效"""
        print("测试缓存失效...")
        
        # 第一次翻译
        self.translator('hello')
        
        # 修改翻译
        self.translator.add_translation('hello', 'zh', '你好（修改后）')
        
        # 应该返回新翻译
        self.assertEqual(self.translator('hello'), '你好（修改后）')
        
        # 验证缓存已更新为新值
        cache_key = 'hello_zh'
        self.assertIn(cache_key, self.translator._cache)
        self.assertEqual(self.translator._cache[cache_key], '你好（修改后）')
        
        print("✅ 缓存失效测试通过")
    
    def test_get_default_instance(self):
        """测试获取默认实例"""
        print("测试获取默认实例...")
        
        # 获取默认实例
        default_translator = Translator.get_default()
        
        # 验证是同一个实例
        self.assertIs(default_translator, Translator.get_default())
        
        # 验证默认语言
        self.assertEqual(default_translator.to_lang, 'en')
        
        print("✅ 获取默认实例测试通过")
    
    def test_set_dictionary(self):
        """测试设置字典"""
        print("测试设置字典...")
        
        new_dict = {
            'new_hello': {
                'zh': '新的你好',
                'en': 'New Hello'
            }
        }
        
        self.translator.set_dictionary(new_dict)
        
        # 验证字典已更新
        self.assertEqual(self.translator.dictionary, new_dict)
        
        # 验证缓存已清除
        self.assertEqual(len(self.translator._cache), 0)
        
        # 验证新翻译工作正常
        self.assertEqual(self.translator('new_hello'), '新的你好')
        
        print("✅ 设置字典测试通过")
    
    def test_get_dictionary(self):
        """测试获取字典"""
        print("测试获取字典...")
        
        dictionary = self.translator.get_dictionary()
        
        # 验证返回的是同一个字典
        self.assertIs(dictionary, self.translator.dictionary)
        
        # 验证内容
        self.assertIn('hello', self.translator.dictionary)
        self.assertIn('world', self.translator.dictionary)
        
        print("✅ 获取字典测试通过")


class TestTranslateFunction(unittest.TestCase):
    """测试translate函数"""
    
    def setUp(self):
        """测试前的设置"""
        self.test_dictionary = {
            'hello': {
                'zh': '你好',
                'en': 'Hello'
            }
        }
    
    def test_translate_with_default_translator(self):
        """测试使用默认翻译器"""
        print("测试使用默认翻译器...")
        
        # 使用默认翻译器
        result = translate('average_reward', 'zh')
        
        # 应该返回翻译结果
        self.assertEqual(result, '平均奖励')
        
        print("✅ 默认翻译器测试通过")
    
    def test_translate_with_custom_dictionary(self):
        """测试使用自定义字典"""
        print("测试使用自定义字典...")
        
        result = translate('hello', 'zh', self.test_dictionary)
        
        # 应该返回翻译结果
        self.assertEqual(result, '你好')
        
        print("✅ 自定义字典测试通过")
    
    def test_translate_with_translator_instance(self):
        """测试使用翻译器实例"""
        print("测试使用翻译器实例...")
        
        translator = Translator(to_lang='zh', dictionary=self.test_dictionary)
        
        result = translate('hello', 'zh', translator)
        
        # 应该返回翻译结果
        self.assertEqual(result, '你好')
        
        print("✅ 翻译器实例测试通过")
    
    def test_translate_invalid_dictionary(self):
        """测试无效字典类型"""
        print("测试无效字典类型...")
        
        with self.assertRaises(ValueError):
            translate('hello', 'zh', 123)
        
        print("✅ 无效字典类型测试通过")


class TestCachedTranslateFunction(unittest.TestCase):
    """测试缓存翻译函数"""
    
    def test_cached_translate(self):
        """测试缓存翻译功能"""
        print("测试缓存翻译功能...")
        
        # 第一次调用
        result1 = _cached_translate('average_reward', 'zh', 0)
        
        # 第二次调用（应该使用缓存）
        result2 = _cached_translate('average_reward', 'zh', 0)
        
        # 结果应该相同
        self.assertEqual(result1, result2)
        self.assertEqual(result1, '平均奖励')
        
        print("✅ 缓存翻译功能测试通过")


class TestTranslatorPerformance(unittest.TestCase):
    """测试翻译器性能"""
    
    def test_translation_performance(self):
        """测试翻译性能"""
        print("测试翻译性能...")
        
        # 创建大字典
        large_dict = {}
        for i in range(1000):
            key = f'key_{i}'
            large_dict[key] = {
                'zh': f'中文翻译_{i}',
                'en': f'English translation_{i}'
            }
        
        translator = Translator(to_lang='zh', dictionary=large_dict)
        
        # 测试重复翻译性能
        start_time = time.time()
        
        for _ in range(1000):
            translator('key_1')
            translator('key_100')
            translator('key_500')
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 验证性能合理（应该在1秒内完成）
        self.assertLess(duration, 1.0, f"翻译性能测试耗时过长: {duration:.4f}秒")
        
        print(f"✅ 翻译性能测试通过，耗时: {duration:.4f}秒")


class TestTranslatorEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_empty_key(self):
        """测试空键"""
        print("测试空键...")
        
        translator = Translator()
        
        # 空键应该返回原键
        self.assertEqual(translator(''), '')
        
        print("✅ 空键测试通过")
    
    def test_none_key(self):
        """测试None键"""
        print("测试None键...")
        
        translator = Translator()
        
        # None键应该返回原键
        self.assertEqual(translator(None), None)
        
        print("✅ None键测试通过")
    
    def test_special_characters(self):
        """测试特殊字符"""
        print("测试特殊字符...")
        
        translator = Translator(to_lang='zh')  # 设置默认语言为中文
        
        # 添加包含特殊字符的翻译
        translator.add_translation('special_key', 'zh', '特殊字符：!@#$%^&*()')
        translator.add_translation('special_key', 'en', 'Special chars: !@#$%^&*()')
        
        # 验证翻译
        self.assertEqual(translator('special_key'), '特殊字符：!@#$%^&*()')
        self.assertEqual(translator('special_key', 'en'), 'Special chars: !@#$%^&*()')
        
        print("✅ 特殊字符测试通过")


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始运行翻译器测试...\n")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestTranslator,
        TestTranslateFunction,
        TestCachedTranslateFunction,
        TestTranslatorPerformance,
        TestTranslatorEdgeCases
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果
    print(f"\n📊 测试结果统计:")
    print(f"   运行测试: {result.testsRun}")
    print(f"   失败测试: {len(result.failures)}")
    print(f"   错误测试: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("🎉 所有测试通过！")
    else:
        print("❌ 有测试失败，请检查错误信息")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    run_all_tests()
