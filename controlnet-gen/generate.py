# controlnet-gen/generate.py
import sys
import os
from pathlib import Path

# Добавляем корневую папку для импорта config
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config

import torch
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import time

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image

def create_output_folder(prompt, base_dir="controlnet_output"):
    """Создает папку для результатов"""
    safe_prompt = "".join(
        c for c in prompt[:50] 
        if c.isalnum() or c in " -_"
    ).strip().replace(" ", "_")
    
    if len(safe_prompt) > 50:
        safe_prompt = safe_prompt[:50]
    
    if not safe_prompt:
        safe_prompt = "unnamed"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{safe_prompt}"
    folder_path = os.path.join(base_dir, folder_name)
    
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def save_prompt_info(folder_path, prompt, num_files, additional_params=None):
    """Сохраняет информацию о промте"""
    info_file = os.path.join(folder_path, "prompt_info.txt")
    
    with open(info_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ControlNet - Информация о генерации\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Промт: {prompt}\n")
        f.write(f"Количество файлов: {num_files}\n")
        
        if additional_params:
            f.write("\nДополнительные параметры:\n")
            for key, value in additional_params.items():
                f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Модель: lllyasviel/control_v11p_sd15_canny\n")
        f.write(f"Устройство: GPU ({torch.cuda.get_device_name(0)})\n")

def get_int_input(prompt_text, default=None, min_val=1, max_val=100, explanation=""):
    """Запрашивает целое число"""
    if explanation:
        print(f"  💡 {explanation}")
    
    while True:
        try:
            user_input = input(prompt_text).strip()
            if not user_input and default is not None:
                return default
            
            value = int(user_input)
            
            if min_val <= value <= max_val:
                return value
            else:
                print(f"  ⚠️  Значение должно быть от {min_val} до {max_val}")
        except ValueError:
            print("  ⚠️  Введите целое число!")

def prepare_control_image(image_path, low_threshold=100, high_threshold=200):
    """Подготовка контрольного изображения (Canny edge detection)"""
    # Загрузка изображения
    if image_path.startswith("http"):
        image = load_image(image_path)
    else:
        image = Image.open(image_path).convert("RGB")
    
    # Конвертация в numpy
    image_np = np.array(image)
    
    # Canny edge detection
    edges = cv2.Canny(image_np, low_threshold, high_threshold)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    
    control_image = Image.fromarray(edges)
    return control_image

def main():
    print("=" * 70)
    print("🎨 ControlNet - Генератор изображений по контуру")
    print("=" * 70)
    
    # Проверка GPU
    if not torch.cuda.is_available():
        print("❌ GPU не найден!")
        return
    
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"✓ PyTorch: {torch.__version__}")
    
    # Очистка
    torch.cuda.empty_cache()
    
    # Загрузка модели
    print("\n📥 Загрузка ControlNet модели...")
    print("   Это может занять 1-2 минуты")
    
    try:
        checkpoint = "lllyasviel/control_v11p_sd15_canny"
        
        controlnet = ControlNetModel.from_pretrained(
            checkpoint, 
            torch_dtype=torch.float16
        )
        
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=controlnet, 
            torch_dtype=torch.float16,
            safety_checker=None
        )
        
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        
        print("✓ Модель загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    # Создание базовой папки
    base_output_dir = Path(__file__).parent / "controlnet_output"
    base_output_dir.mkdir(exist_ok=True)
    
    while True:
        print("\n" + "=" * 70)
        print("СОЗДАНИЕ ИЗОБРАЖЕНИЯ ПО КОНТУРУ")
        print("=" * 70)
        
        # Источник контрольного изображения
        print("\n📁 ИСХОДНОЕ ИЗОБРАЖЕНИЕ")
        print("   1. Использовать пример из интернета")
        print("   2. Указать путь к файлу")
        
        source_choice = input("   Выбор (1-2, Enter=1): ").strip()
        
        if source_choice == "2":
            print("   Введите путь без кавычек!")
            image_path = input("   Путь: ").strip()
            # Удаляем кавычки, если пользователь их ввел
            image_path = image_path.strip('"').strip("'")
            if not image_path:
                image_path = "https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/images/input.png"
        else:
            image_path = "https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/images/input.png"
            
        # Промт
        print("\n📝 ОПИСАНИЕ")
        print("   Опишите, что хотите создать на основе контура.")
        prompt = input("🎨 Введите описание: ").strip()
        
        if prompt.lower() == 'q':
            break
        
        if not prompt:
            prompt = "a blue paradise bird in the jungle"
            print(f"   Используется промт по умолчанию: {prompt}")
        
        # Пороги Canny
        print("\n⚙️  ПАРАМЕТРЫ CANNY")
        low_threshold = get_int_input(
            "   Нижний порог (Enter=100): ",
            default=100,
            min_val=0,
            max_val=255,
            explanation="Чем ниже, тем больше деталей"
        )
        
        high_threshold = get_int_input(
            "   Верхний порог (Enter=200): ",
            default=200,
            min_val=0,
            max_val=255,
            explanation="Чем выше, тем меньше шума"
        )
        
        # Количество файлов
        print("\n📁 КОЛИЧЕСТВО")
        num_files = get_int_input(
            "   Сколько изображений создать? (Enter=1): ",
            default=1,
            min_val=1,
            max_val=20000,
            explanation="Каждое будет уникальной вариацией"
        )
        
        # Шаги
        print("\n⚙️  КАЧЕСТВО")
        num_steps = get_int_input(
            "   Количество шагов (Enter=20): ",
            default=20,
            min_val=10,
            max_val=500,
            explanation="Больше шагов = лучше качество"
        )
        
        # Подготовка контрольного изображения
        print("\n🔄 Подготовка контрольного изображения...")
        try:
            control_image = prepare_control_image(
                image_path, 
                low_threshold, 
                high_threshold
            )
            print("✓ Контрольное изображение готово")
        except Exception as e:
            print(f"❌ Ошибка подготовки: {e}")
            continue
        
        # Создание папки
        output_folder = create_output_folder(prompt, str(base_output_dir))
        print(f"\n📂 Создана папка: {output_folder}")
        
        # Сохранение контрольного изображения
        control_path = os.path.join(output_folder, "control_image.png")
        control_image.save(control_path)
        print(f"💾 Контрольное изображение: {control_path}")
        
        # Параметры
        additional_params = {
            "num_inference_steps": num_steps,
            "canny_low_threshold": low_threshold,
            "canny_high_threshold": high_threshold,
            "model": "lllyasviel/control_v11p_sd15_canny",
        }
        
        # Сохранение информации
        save_prompt_info(output_folder, prompt, num_files, additional_params)
        print("📄 Сохранен prompt_info.txt")
        
        print(f"\n🎨 Генерация: {prompt}")
        print(f"📁 Количество: {num_files}")
        print(f"⚙️ Шагов: {num_steps}")
        print("\n⏳ Ожидайте...")
        
        # Генерация
        successful = 0
        total_time = 0
        
        for i in range(1, num_files + 1):
            print(f"\n[{i}/{num_files}] Генерация...")
            
            start_time = time.time()
            
            try:
                # Уникальный seed
                seed = int(time.time() * 1000) + i * 1000
                generator = torch.Generator("cuda").manual_seed(seed)
                
                # Генерация
                image = pipe(
                    prompt,
                    num_inference_steps=num_steps,
                    generator=generator,
                    image=control_image,
                ).images[0]
                
                elapsed = time.time() - start_time
                total_time += elapsed
                
                # Сохранение
                output_path = os.path.join(output_folder, f"generated_{i:03d}.png")
                image.save(output_path)
                
                successful += 1
                
                file_size = os.path.getsize(output_path) / 1024
                print(f"  ✅ generated_{i:03d}.png ({file_size:.0f} KB, {elapsed:.1f} сек)")
                
                # Очистка
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ❌ Ошибка: {e}")
                torch.cuda.empty_cache()
        
        # Итоги
        print(f"\n{'=' * 70}")
        print("✅ ГОТОВО")
        print(f"{'=' * 70}")
        print(f"📁 Успешно: {successful}/{num_files}")
        if successful > 0:
            print(f"⏱️ Среднее время: {total_time/successful:.1f} сек/изображение")
        print(f"📂 Папка: {output_folder}")
        
        cont = input("\nСоздать еще? (y/n): ").strip().lower()
        if cont != 'y':
            break
    
    print("\n👋 До свидания!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()