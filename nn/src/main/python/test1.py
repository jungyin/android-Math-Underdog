import re
import json

def process_mixed_content(full_string):
    """
    处理包含特殊标签和数学公式的混合字符串。

    Args:
        full_string (str): 包含 <s><s>, </s> 和 \[ \] 数学公式的字符串。

    Returns:
        list: 一个列表，每个元素是一个字典，表示文本块或公式块，
              例如：[{"type": "text", "content": "..."}] 或
              [{"type": "formula", "content": "..."}]
    """
    # 1. 移除最外层的 <s><s> 和 </s> 标签
    cleaned_string = full_string.replace("<s><s>", "").replace("</s>", "").strip()

    # 2. 定义正则表达式来匹配数学公式块（包括其前后的空白符），并捕获公式内容
    # re.DOTALL 确保 . 匹配包括换行符在内的所有字符
    # 外层捕获组 () 是为了让 re.split 保留分隔符本身
    math_formula_pattern = r'(\s*\\\[.*?\\\]\s*)'

    # 使用 re.split() 分割文本，因为模式有捕获组，所以匹配到的分隔符也会保留在结果列表中
    parts = re.split(math_formula_pattern, cleaned_string, flags=re.DOTALL)

    results = []
    for part in parts:
        part_stripped = part.strip() # 先去除当前片段的空白符

        if not part_stripped:
            continue # 跳过完全空白的片段

        # 判断是否是数学公式块
        if part_stripped.startswith('\\[') and part_stripped.endswith('\\]'):
            # 这是一个数学公式，移除 \[ 和 \]
            formula_content = part_stripped[2:-2].strip()
            results.append({"type": "formula", "content": formula_content})
        else:
            # 这是普通文本
            results.append({"type": "text", "content": part_stripped})

    return results

# 您的原始字符串
text_data = """<s><s>
 \[
\frac{1\prod_{j,k}^{p}[p,L1]} = \frac{t_{j,k+\widetilde p-1}-t_{j,k+1}}{t_{j,k+\widetilde p}-t_{j,k}} \stackrel{\*}{\* d*j,k[i]}\,,
 \]
otocols that employ the XOR operator can be modeled by th
 \[
\mathrm{eu}\,\,\mathbb{H}^{*}\left(S_{-d}^{3}(K),a\right)=-\sum_{\substack{j\equiv a(\mathrm{mod{ d}})\\ 0\leq j\leq M}}\mathrm{~eu}\,\,\mathbb{H}^{*}\left(T_{j},W\right).
 \]</s>"""

processed_content = process_mixed_content(text_data)

print(json.dumps(processed_content, indent=2, ensure_ascii=False))

# 遍历并打印结果
print("\n--- 遍历处理后的内容 ---")
for item in processed_content:
    if item['type'] == 'text':
        print(f"文本: {item['content']}")
    elif item['type'] == 'formula':
        print(f"公式: {item['content']}")