# music-gen/generate.py
import sys
import os
from pathlib import Path

# Добавляем корневую папку в sys.path для импорта config
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Импортируем конфигурацию (автоматически установит переменные окружения)
import config

import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import numpy as np
import time
from datetime import datetime

def create_output_folder(prompt, base_dir="music_output"):
    """Создает папку для музыкальных файлов"""
    safe_prompt = "".join(
        c for c in prompt[:50] 
        if c.isalnum() or c in " -_"
    ).strip().replace(" ", "_")
    
    if len(safe_prompt) > 50:
        safe_prompt = safe_prompt[:50]
    
    if not safe_prompt:
        safe_prompt = "unnamed_music"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{safe_prompt}"
    folder_path = os.path.join(base_dir, folder_name)
    
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def save_prompt_info(folder_path, prompt, duration, additional_params=None):
    """Сохраняет информацию о промте"""
    info_file = os.path.join(folder_path, "prompt_info.txt")
    
    with open(info_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("MusicGen Large - Информация о генерации\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Промт: {prompt}\n")
        f.write(f"Длительность: {duration} сек\n")
        
        if additional_params:
            f.write("\nДополнительные параметры:\n")
            for key, value in additional_params.items():
                f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Устройство: GPU ({torch.cuda.get_device_name(0)})\n")
        f.write(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

def get_int_input(prompt_text, default=10, min_val=5, max_val=60, explanation=""):
    """Запрашивает целое число с пояснением"""
    if explanation:
        print(f"  💡 {explanation}")
    
    while True:
        try:
            user_input = input(prompt_text).strip()
            if not user_input:
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
    print("🎶 MusicGen Large - Генератор музыки")
    print("=" * 70)
    
    # Проверка GPU
    if not torch.cuda.is_available():
        print("❌ GPU не найден!")
        return
    
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"✓ PyTorch: {torch.__version__}")
    
    # Очистка кэша
    torch.cuda.empty_cache()
    
    # Загрузка модели
    print("\n📥 Загрузка модели MusicGen Large...")
    print("   Это может занять 30-60 секунд")
    
    try:
        processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-large",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("cuda")
        
        print("✓ Модель загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    # Создание базовой папки для результатов
    base_output_dir = Path(__file__).parent / "music_output"
    base_output_dir.mkdir(exist_ok=True)
    
    # Интерактивный режим
    while True:
        print("\n" + "=" * 70)
        print("СОЗДАНИЕ МУЗЫКАЛЬНОЙ КОМПОЗИЦИИ")
        print("=" * 70)
        
        # Промт
        print("\n📝 ОПИСАНИЕ МУЗЫКИ")
        print("   Опишите музыку, которую хотите создать.")
        print("   Примеры: 'lo-fi music', 'epic orchestral', 'jazz piano'")
        prompt = input("🎵 Введите описание: ").strip()
        
        if prompt.lower() == 'q':
            break
        
        if not prompt:
            prompt = "lo-fi music with a soothing melody"
            print(f"   Используется промт по умолчанию: {prompt}")
        
        # Длительность
        print("\n⏱️  ДЛИТЕЛЬНОСТЬ")
        duration = get_int_input(
            "   Введите длительность (сек, Enter=10): ",
            default=10,
            min_val=5,
            max_val=60,
            explanation="Рекомендуется 10-30 секунд"
        )
        
        # Количество вариаций
        print("\n📁 КОЛИЧЕСТВО ФАЙЛОВ")
        num_files = get_int_input(
            "   Сколько файлов создать? (Enter=1): ",
            default=1,
            min_val=1,
            max_val=30,
            explanation="Каждый файл будет уникальной вариацией"
        )
        
        # Создание папки
        output_folder = create_output_folder(prompt, str(base_output_dir))
        print(f"\n📂 Создана папка: {output_folder}")
        
        # Параметры
        additional_params = {
            "model": "facebook/musicgen-large",
            "guidance_scale": 3.0,
            "temperature": 1.0,
        }
        
        # Сохранение информации
        save_prompt_info(output_folder, prompt, duration, additional_params)
        print("📄 Сохранен prompt_info.txt")
        
        print(f"\n🎵 Генерация: {prompt}")
        print(f"⏱️  Длительность: {duration} сек")
        print(f"📁 Количество файлов: {num_files}")
        print("\n⏳ Ожидайте, идет генерация...")
        
        # Подготовка входных данных (одинаковые для всех)
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        
        max_new_tokens = duration * 50  # 50 токенов в секунду
        
        # Генерация файлов
        successful = 0
        total_time = 0
        
        for i in range(1, num_files + 1):
            print(f"\n[{i}/{num_files}] Генерация...")
            
            start_time = time.time()
            
            try:
                # Уникальный seed для каждой вариации
                seed = int(time.time() * 1000) + i * 1000
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                
                # Генерация
                with torch.no_grad():
                    audio_values = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        guidance_scale=3.0,
                        temperature=1.0,
                    )
                
                elapsed = time.time() - start_time
                total_time += elapsed
                
                # Сохранение
                sampling_rate = model.config.audio_encoder.sampling_rate
                output_path = os.path.join(output_folder, f"variation_{i:03d}.wav")
                
                # Конвертация
                audio_data = audio_values[0, 0].cpu().numpy()
                
                if audio_data.dtype == np.float16:
                    audio_data = audio_data.astype(np.float32)
                
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                audio_int16 = (audio_data * 32767).astype(np.int16)
                
                scipy.io.wavfile.write(
                    output_path,
                    rate=sampling_rate,
                    data=audio_int16
                )
                
                successful += 1
                
                file_size = os.path.getsize(output_path) / 1024 / 1024
                print(f"  ✅ variation_{i:03d}.wav ({file_size:.1f} MB, {elapsed:.1f} сек)")
                
                # Обновление информации
                with open(os.path.join(output_folder, "prompt_info.txt"), "a", encoding="utf-8") as f:
                    f.write(f"\nФайл {i}: variation_{i:03d}.wav\n")
                    f.write(f"  Seed: {seed}\n")
                    f.write(f"  Время генерации: {elapsed:.1f} сек\n")
                    f.write(f"  Размер: {file_size:.1f} MB\n")
                
                # Очистка кэша
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ❌ Ошибка: {e}")
                torch.cuda.empty_cache()
        
        # Итоги
        print(f"\n{'=' * 70}")
        print("✅ ГЕНЕРАЦИЯ ЗАВЕРШЕНА")
        print(f"{'=' * 70}")
        print(f"📁 Успешно: {successful}/{num_files}")
        print(f"⏱️  Общее время: {total_time:.1f} сек ({total_time/60:.1f} мин)")
        print(f"📂 Папка: {output_folder}")
        
        # Очистка
        torch.cuda.empty_cache()
        
        # Продолжение
        print("\n" + "=" * 70)
        continue_choice = input("Создать еще музыку? (y/n): ").strip().lower()
        if continue_choice != 'y':
            break
    
    print("\n👋 До свидания!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()