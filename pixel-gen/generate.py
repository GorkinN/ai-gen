import sys
import os
from pathlib import Path

# Добавляем корневую папку для импорта config
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config

import torch
import functools
# Обход для старых моделей с .bin файлами
torch.load = functools.partial(torch.load, weights_only=False)

from diffusers import StableDiffusionPipeline
import time
from datetime import datetime

def create_output_folder(prompt, base_dir="pixel_output"):
    """Создает папку для пиксельных изображений"""
    safe_prompt = "".join(
        c for c in prompt[:50] 
        if c.isalnum() or c in " -_"
    ).strip().replace(" ", "_")
    
    if len(safe_prompt) > 50:
        safe_prompt = safe_prompt[:50]
    
    if not safe_prompt:
        safe_prompt = "unnamed_pixel"
    
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
        f.write("Pixel Art Generator - Информация о генерации\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Промт: {prompt}\n")
        f.write(f"Количество файлов: {num_files}\n")
        
        if additional_params:
            f.write("\nДополнительные параметры:\n")
            for key, value in additional_params.items():
                f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Модель: Onodofthenorth/SD_PixelArt_SpriteSheet_Generator\n")
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

def main():
    print("=" * 70)
    print("👾 Pixel Art Generator - Генератор пиксельных изображений")
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
    print("\n📥 Загрузка Pixel Art модели...")
    print("   Это может занять 1-2 минуты")
    
    try:
        model_id = "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
        
        print("✓ Модель загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    # Создание базовой папки
    base_output_dir = Path(__file__).parent / "pixel_output"
    base_output_dir.mkdir(exist_ok=True)
    
    while True:
        print("\n" + "=" * 70)
        print("СОЗДАНИЕ ПИКСЕЛЬНОГО ИЗОБРАЖЕНИЯ")
        print("=" * 70)
        
        # Промт
        print("\n📝 ОПИСАНИЕ")
        print("   Опишите, что хотите создать в пиксельном стиле.")
        print("   Примеры: 'warrior', 'treasure chest', 'forest scene'")
        print("   Или 'PixelartLSS' для спрайт-листа")
        prompt = input("👾 Введите описание: ").strip()
        
        if prompt.lower() == 'q':
            break
        
        if not prompt:
            prompt = "PixelartLSS"
            print(f"   Используется промт по умолчанию: {prompt}")
        
        # Количество файлов
        print("\n📁 КОЛИЧЕСТВО")
        num_files = get_int_input(
            "   Сколько изображений создать? (Enter=1): ",
            default=1,
            min_val=1,
            max_val=20,
            explanation="Каждое изображение будет уникальной вариацией"
        )
        
        # Количество шагов
        print("\n⚙️  КАЧЕСТВО")
        num_steps = get_int_input(
            "   Количество шагов (Enter=30): ",
            default=30,
            min_val=15,
            max_val=500,
            explanation="Больше шагов = лучше качество"
        )
        
        # Создание папки
        output_folder = create_output_folder(prompt, str(base_output_dir))
        print(f"\n📂 Создана папка: {output_folder}")
        
        # Параметры
        additional_params = {
            "num_inference_steps": num_steps,
            "model": "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator",
        }
        
        # Сохранение информации
        save_prompt_info(output_folder, prompt, num_files, additional_params)
        print("📄 Сохранен prompt_info.txt")
        
        print(f"\n👾 Генерация: {prompt}")
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
                ).images[0]
                
                elapsed = time.time() - start_time
                total_time += elapsed
                
                # Сохранение
                output_path = os.path.join(output_folder, f"pixel_{i:03d}.png")
                image.save(output_path)
                
                successful += 1
                
                file_size = os.path.getsize(output_path) / 1024
                print(f"  ✅ pixel_{i:03d}.png ({file_size:.0f} KB, {elapsed:.1f} сек)")
                
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