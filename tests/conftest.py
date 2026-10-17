"""tests/conftest.py — 用 importlib 加载数字开头的示例脚本模块"""
import os
import sys

import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_module(filename):
    """加载仓库根目录下的示例脚本 (文件名以数字开头, 不能常规 import)"""
    path = os.path.join(ROOT, filename)
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
