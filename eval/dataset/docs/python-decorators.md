# Python 装饰器笔记

装饰器本质上是一个接收函数并返回函数的可调用对象。基本写法：

```python
import functools

def retry(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

要点：
- 一定要用 `@functools.wraps` 保留原函数的 `__name__` 和 docstring
- 带参数的装饰器需要三层嵌套：参数层 → 装饰器层 → 包装层
- 类装饰器实现 `__call__` 即可，适合需要维护状态的场景
- 常见用途：日志、重试、缓存（`functools.lru_cache`）、权限校验
