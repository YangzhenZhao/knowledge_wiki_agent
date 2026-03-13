# Calculator

## 基本信息

- **名称**: calculator
- **描述**: 执行数学计算。当用户问题包含数学运算（加减乘除等）时，必须调用此工具进行精确计算，不要自己估算或心算。
- **触发词**: 计算, 等于, 多少, 算一下

## 参数

| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| expression | string | 是 | 数学表达式，如 `2+3*4` 或 `(10-5)/2` |

## 示例

**用户**: 帮我计算 123 * 456 等于多少
**调用**: `calculator(expression="123 * 456")`
**返回**: 🧮 计算结果: 123 * 456 = 56088

## 执行逻辑

```python
def execute(expression: str) -> str:
    allowed = set("0123456789+-*/.()^ eEsSiInNpPlLoOgGtTcCsSqrtaAbB ")
    if not all(c in allowed for c in expression):
        return f"表达式包含不允许的字符"

    expr = expression.replace("^", "**")
    safe_dict = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log10": math.log10,
        "exp": math.exp, "pi": math.pi, "e": math.e,
        "abs": abs, "pow": pow,
    }
    result = eval(expr, {"__builtins__": {}}, safe_dict)

    if isinstance(result, float):
        result = int(result) if result.is_integer() else round(result, 10)

    return f"🧮 计算结果: {expression} = {result}"
```

## 注意事项

- 只支持数学运算，不能执行任意代码
- 支持常用数学函数: sqrt, sin, cos, tan, log, exp, pow
- 常量: pi, e