# AI Coding Agent编程风格模板：Python 3.10+

## 🎯 核心原则

1. **严格遵循PEP 8标准**：所有代码的缩进、空格、行长度、导入顺序等必须符合此标准。
2. **严格遵循命名约定**：类名使用PascalCase，函数和方法名使用snake_case，常量使用UPPER_SNAKE_CASE。
3. **严格遵循完整描述性命名规范**：类名、函数名、方法名、常量名均使用使用单词全称，避免使用缩写或简写，专有名词除外。
4. **必须编写核心功能测试**：所有核心功能模块都必须包含测试样例。
5. **必须添加类型提示**：所有函数和方法的参数和返回值都必须添加类型提示。
6. **必须编写文档字符串**：所有共有模块、类、函数和方法都必须包含Google风格的文档字符串。
7. **优先保证可读性**：代码结构应当简洁清晰，如同自然语言一样便于阅读和理解。

---

## 📛 命名规范

| 类型 | 规则 | 示例 |
|------|------|------|
| **类** | `PascalCase` | `ClassName` |
| **常量** | `UPPER_SNAKE_CASE` | `CONSTANT_NAME` |
| **变量/函数/方法** | `snake_case` + 完整描述 | `variable_name`, `function_name()`, `method_name()` |
| **私有成员** | `snake_case` +  `_`前缀 | `_private_variable`, `_private_method()` |
| **异常类** | `snake_case` +  `Error`后缀 | `invalid_xxxx_error` |

---

## 📦 类/函数/方法结构整体规范

```python
# 设置ClassName类
class ClassName:
    def __init__(self, arg1: type1, arg2: type2 = default_value) -> None:
        """
        初始化类实例。
    
        Args:
            arg1: 参数1描述。
            arg2: 参数2描述，包含默认值说明。
        """
        self.attribute_name = arg1
        self._private_attribute_name = arg2

    def public_method(self, parameter1: param_type1, parameter2: param_type2) -> return_type:
        """
        公有方法功能描述。
    
        Args:
            parameter1: 参数1详细描述。
            parameter2: 参数2详细描述。
        
        Returns:
            返回值详细描述。
        
        Raises:
            ExceptionType: 异常情况描述。
        """
        # 方法实现
        result = self._some_method(parameter1, parameter2)
        return result

    def _private_method(self, parameter1: parameter_type1, parameter2: parameter_type2) -> return_type:
        """
        私有方法功能描述。
    
        Args:
            parameter1: 参数1详细描述。
            parameter2: 参数2详细描述。
        
        Returns:
            返回值详细描述。
        
        Raises:
            ExceptionType: 异常情况描述。
        """
        # 方法实现
        result = self._some_method(parameter1, parameter2)
        return result
```

## 🏷️ 常见专有名词规范

| 领域 | 允许使用的专有名词示例 |
| :--- | :--- |
| **网络与协议** | `http`, `https`, `url`, `uri`, `api`, `json`, `xml`, `html`, `css`, `js` |
| **硬件与系统** | `cpu`, `gpu`, `ram`, `rom`, `ssd`, `io` |
| **图像与多媒体** | `rgb`, `hsv`, `yuv`, `gray`, `binary`, `fps`, `ssim`, `psnr` |
| **通用技术与格式** | `id`, `uuid`, `sql`, `csv`, `ssl`, `tls` |