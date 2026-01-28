import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
from pathlib import Path

class ImageAugmentation:
    def __init__(self, input_dir='blur image', output_dir='augmented_images'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ensure_output_dir()
        
    def ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def add_gaussian_noise(self, image, mean=0, sigma=25):
        """添加高斯噪声"""
        noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image
    
    def add_salt_pepper_noise(self, image, prob=0.05):
        """添加椒盐噪声"""
        noisy_image = np.copy(image)
        # 添加盐噪声
        salt_mask = np.random.random(image.shape[:2]) < prob/2
        noisy_image[salt_mask] = 255
        # 添加椒噪声
        pepper_mask = np.random.random(image.shape[:2]) < prob/2
        noisy_image[pepper_mask] = 0
        return noisy_image
    
    def blur_image(self, image, kernel_size=(5,5)):
        """高斯模糊"""
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    def adjust_brightness(self, image, factor_range=(0.5, 1.5)):
        """调整亮度"""
        factor = random.uniform(*factor_range)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_image)
        brightened_image = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(brightened_image), cv2.COLOR_RGB2BGR)
    
    def rotate_image(self, image, angle_range=(-30, 30)):
        """随机旋转"""
        angle = random.uniform(*angle_range)
        height, width = image.shape[:2]
        center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image
    
    def random_crop(self, image, crop_ratio_range=(0.8, 1.0)):
        """随机裁剪"""
        height, width = image.shape[:2]
        crop_ratio = random.uniform(*crop_ratio_range)
        new_height = int(height * crop_ratio)
        new_width = int(width * crop_ratio)
        
        start_x = random.randint(0, width - new_width)
        start_y = random.randint(0, height - new_height)
        
        cropped_image = image[start_y:start_y+new_height, start_x:start_x+new_width]
        return cv2.resize(cropped_image, (width, height))
    
    def apply_random_augmentation(self, image):
        """随机应用一种数据增强方法"""
        augmentation_methods = [
            (self.add_gaussian_noise, {}),
            (self.add_salt_pepper_noise, {}),
            (self.blur_image, {}),
            (self.adjust_brightness, {}),
            (self.rotate_image, {}),
            (self.random_crop, {})
        ]
        
        # 随机选择一种方法
        method, params = random.choice(augmentation_methods)
        return method(image, **params)
    
    def process_image(self, image_path):
        """处理单张图片"""
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图片: {image_path}")
            return
        
        # 获取文件名（不含扩展名）
        filename = Path(image_path).stem
        
        # 应用数据增强
        augmented_image = self.apply_random_augmentation(image)
        
        # 保存增强后的图片
        output_path = os.path.join(self.output_dir, f"{filename}.jpg")
        cv2.imwrite(output_path, augmented_image)
        
    def process_directory(self):
        """处理整个目录的图片"""
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        
        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(self.input_dir, filename)
                self.process_image(image_path)
                print(f"已处理: {filename}")

def main():
    # 创建增强器实例
    augmentor = ImageAugmentation()
    
    # 处理所有图片
    print("开始处理图片...")
    augmentor.process_directory()
    print("处理完成！")

if __name__ == "__main__":
    main() 