# 文档主标题 (H1)

这是引言段落，介绍本文档的主要内容。包含一些**加粗文本**和*斜体文本*。

## 第一章节：核心概念 (H2)

本章节将详细讨论核心概念。

### 1.1 概念A (H3)

概念A的详细描述。这是一个有序列表：
1. 第一个要点
2. 第二个要点
   * 子要点2.1 (嵌套无序列表)
   * 子要点2.2
3. 第三个要点

### 1.2 概念B (H3)

概念B的详细描述。这是一个无序列表：
- 项目符号1
- 项目符号2
  - 嵌套项目符号2.a
  - 嵌套项目符号2.b

## 第二章节：数据表格与代码 (H2)

下面是一个简单的Markdown表格：

| 头部1 | 头部2 | 头部3 |
|-------|-------|-------|
| 单元1 | 单元2 | 单元3 |
| 单元4 | 单元5 | 单元6 |

以及一个代码块：

```python
def hello_world():
print("Hello, Markdown!")
```

> 这是一个块引用。
> 包含多行。

文档末尾。


| Header 1 | Header 2 | Header 3 | Long Description Column                                       |
|----------|----------|----------|---------------------------------------------------------------|
| Data 1A  | Data 1B  | Data 1C  | This is a very long description for the first row, it contains many words and details to make the row exceed typical length. We need to ensure this entire description, when combined with other columns, makes the markdown representation of this table row quite substantial. More text here to fill it up. |
| Data 2A  | Data 2B  | Data 2C  | Another lengthy description for the second row, designed to test the splitting capabilities when individual rows contribute significantly to the overall size of the markdown table. We are adding more and more verbose content. |
| Data 3A  | Data 3B  | Data 3C  | Short one.                                                    |
| Data 4A  | Data 4B  | Data 4C  | Yet another extensive narration for the fourth data entry, pushing the boundaries of our chunking mechanism specifically for table elements. The goal is to see how well it handles multiple long rows and splits them appropriately while retaining context via headers. |
| Data 5A  | Data 5B  | Data 5C  | This row also has a considerable amount of text. It's important that the table splitter correctly identifies row boundaries and includes the header with each generated chunk. We are still adding more text to make it longer. |
| Data 6A  | Data 6B  | Data 6C  | The sixth row continues the pattern of providing detailed textual information that contributes to the overall length of the table, thereby testing the robustness of our row-based splitting logic. This should be enough. |
| Data 7A  | Data 7B  | Data 7C  | Penultimate row with a lot of information to ensure that we have enough content to trigger the max length for splitting if the target size is met by a few rows. This is just filler text repeating for length. This is just filler text repeating for length. |
| Data 8A  | Data 8B  | Data 8C  | The final row, also verbose, to complete this large table example. It will be interesting to see how the last set of rows is chunked, especially if it forms a chunk smaller than the target but larger than a minimum. |