import os
import base64
from PIL import Image
from io import BytesIO


def images_to_tsv(input_folder, output_file):
    with open(output_file, 'w', encoding='utf8') as tsv:
        # 统计成功和失败的数量
        success_count = 0
        fail_count = 0

        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img_path = os.path.join(input_folder, filename)
                try:
                    with Image.open(img_path) as img:
                        img_buffer = BytesIO()
                        # 使用更可靠的格式处理方式
                        format = img.format or filename.split('.')[-1].upper()
                        img.save(img_buffer, format=format)
                        byte_data = img_buffer.getvalue()
                        base64_str = base64.b64encode(byte_data).decode('utf8')

                    img_id = os.path.splitext(filename)[0]
                    tsv.write(f"{img_id}\t{base64_str}\n")
                    success_count += 1
                except Exception as e:
                    # 输出详细的错误信息
                    print(f"处理文件 {filename} 时出错: {str(e)}")
                    fail_count += 1

        print(f"处理完成，成功: {success_count} 个，失败: {fail_count} 个")


# 使用示例
images_to_tsv("new image", "output.tsv")