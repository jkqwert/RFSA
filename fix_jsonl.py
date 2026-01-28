import json


def fix_jsonl_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                # 尝试解析JSON
                json.loads(line)
                # 解析成功，写入新文件
                f_out.write(line + '\n')
            except json.JSONDecodeError as e:
                print(f"Line {line_num} is invalid: {e}")


# 使用示例
fix_jsonl_file('Dataloader/datasets/new/test_texts.jsonl', 'Dataloader/datasets/new/test_texts.jsonl')