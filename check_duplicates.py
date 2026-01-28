import json
import os
from collections import defaultdict

def check_duplicates(file_path):
    print(f"\n检查文件: {file_path}")
    
    # 用于存储所有ID
    text_ids = set()
    image_ids = set()
    
    # 用于存储重复的ID
    duplicate_text_ids = defaultdict(list)
    duplicate_image_ids = defaultdict(list)
    
    # 用于存储text_id和image_ids的映射关系
    text_to_images = defaultdict(list)
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                text_id = data.get('text_id')
                image_ids_list = data.get('image_ids', [])
                
                # 检查text_id是否重复
                if text_id in text_ids:
                    duplicate_text_ids[text_id].append(line_num)
                else:
                    text_ids.add(text_id)
                
                # 检查image_ids是否重复
                for img_id in image_ids_list:
                    if img_id in image_ids:
                        duplicate_image_ids[img_id].append((text_id, line_num))
                    else:
                        image_ids.add(img_id)
                
                # 记录text_id和image_ids的映射关系
                text_to_images[text_id].extend(image_ids_list)
                
            except json.JSONDecodeError:
                print(f"警告: 第{line_num}行JSON解析错误")
                continue
    
    # 打印结果
    print("\n=== 重复的text_id ===")
    if duplicate_text_ids:
        for text_id, lines in duplicate_text_ids.items():
            print(f"text_id: {text_id} 在以下行重复出现: {lines}")
    else:
        print("没有重复的text_id")
    
    print("\n=== 重复的image_id ===")
    if duplicate_image_ids:
        for img_id, occurrences in duplicate_image_ids.items():
            print(f"image_id: {img_id} 在以下位置重复出现:")
            for text_id, line_num in occurrences:
                print(f"  - text_id: {text_id}, 行号: {line_num}")
    else:
        print("没有重复的image_id")
    
    print(f"\n总共有 {len(text_ids)} 个唯一的text_id")
    print(f"总共有 {len(image_ids)} 个唯一的image_id")

    return duplicate_text_ids, duplicate_image_ids

def main():
    # 检查所有JSONL文件
    jsonl_files = ['Dataloader/datasets/new/test_texts.jsonl']
    
    for file_name in jsonl_files:
        file_path = os.path.join('', file_name)
        if os.path.exists(file_path):
            check_duplicates(file_path)
        else:
            print(f"文件不存在: {file_path}")

if __name__ == "__main__":
    main() 