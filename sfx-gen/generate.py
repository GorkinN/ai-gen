# sfx-gen/generate.py
import sys
import os
from pathlib import Path

# Добавляем корневую папку в sys.path для импорта config
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Импортируем конфигурацию (она автоматически установит переменные окружения)
import config

# Оптимизация памяти CUDA
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import soundfile as sf
from diffusers import StableAudioPipeline
import time
import numpy as np
from datetime import datetime
import gc

def create_output_folder(prompt, base_dir="sfx_output"):
    """Создает папку для звуковых эффектов"""
    safe_prompt = "".join(
        c for c in prompt[:50] 
        if c.isalnum() or c in " -_"
    ).strip().replace(" ", "_")
    
    if len(safe_prompt) > 50:
        safe_prompt = safe_prompt[:50]
    
    if not safe_prompt:
        safe_prompt = "unnamed_sfx"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{safe_prompt}"
    folder_path = os.path.join(base_dir, folder_name)
    
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def save_prompt_info(folder_path, prompt, duration, num_files, additional_params=None):
    """Сохраняет информацию о промте"""
    info_file = os.path.join(folder_path, "prompt_info.txt")
    
    with open(info_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Stable Audio Open - Информация о генерации\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Промт: {prompt}\n")
        f.write(f"Длительность: {duration:.2f} сек ({duration*1000:.0f} мс)\n")
        f.write(f"Количество файлов: {num_files}\n")
        
        if additional_params:
            f.write("\nДополнительные параметры:\n")
            for key, value in additional_params.items():
                f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Устройство: GPU ({torch.cuda.get_device_name(0)})\n")
        f.write(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

def clear_gpu_memory():
    """Очистка памяти GPU"""
    gc.collect()
    torch.cuda.empty_cache()

def get_float_input(prompt_text, default=1.0, min_val=0.1, max_val=47.0, explanation=""):
    """Запрашивает число с плавающей точкой"""
    if explanation:
        print(f"  💡 {explanation}")
    
    while True:
        try:
            user_input = input(prompt_text).strip()
            if not user_input:
                return default
            
            # Поддержка запятой как разделителя
            user_input = user_input.replace(',', '.')
            value = float(user_input)
            
            if min_val <= value <= max_val:
                return value
            else:
                print(f"  ⚠️  Значение должно быть от {min_val} до {max_val} секунд")
        except ValueError:
            print("  ⚠️  Введите число! (например: 0.5 или 1,5)")

def get_int_input(prompt_text, default=1, min_val=1, max_val=10, explanation=""):
    """Запрашивает целое число"""
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

def generate_single_audio(pipe, prompt, duration, num_steps, seed):
    """Генерация одного аудио файла"""
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # Для очень коротких звуков используем минимальную длительность
    min_duration = 0.5  # Минимальная поддерживаемая длительность
    actual_duration = max(duration, min_duration)
    
    audio = pipe(
        prompt,
        negative_prompt="Low quality, distorted",
        num_inference_steps=num_steps,
        audio_end_in_s=actual_duration,
        num_waveforms_per_prompt=1,  # Всегда 1 для экономии памяти
        generator=generator,
    ).audios
    
    # Обрезаем до нужной длительности
    if duration < min_duration:
        sample_rate = pipe.vae.sampling_rate
        samples_to_keep = int(duration * sample_rate)
        audio_waveform = audio[0]
        
        # Обрезаем по времени (учитывая стерео)
        if audio_waveform.dim() == 2:  # (channels, samples)
            audio_waveform = audio_waveform[:, :samples_to_keep]
        else:  # (samples,)
            audio_waveform = audio_waveform[:samples_to_keep]
        
        return audio_waveform
    
    return audio[0]

def main():
    print("=" * 70)
    print("🎵 Stable Audio Open - Генератор звуковых эффектов")
    print("=" * 70)
    
    # Проверка GPU
    if not torch.cuda.is_available():
        print("❌ GPU не найден!")
        return
    
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"✓ PyTorch: {torch.__version__}")
    
    # Очистка кэша
    clear_gpu_memory()
    
    # Загрузка модели
    print("\n📥 Загрузка модели Stable Audio Open...")
    print("   Это может занять 1-2 минуты")
    
    try:
        pipe = StableAudioPipeline.from_pretrained(
            "stabilityai/stable-audio-open-1.0",
            torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
        
        # Включение оптимизаций памяти
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        
        print("✓ Модель загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    # Создание базовой папки для результатов (в папке скрипта)
    base_output_dir = Path(__file__).parent / "sfx_output"
    base_output_dir.mkdir(exist_ok=True)
    
    # Интерактивный режим
    while True:
        print("\n" + "=" * 70)
        print("СОЗДАНИЕ ЗВУКОВОГО ЭФФЕКТА")
        print("=" * 70)
        
        # Промт
        print("\n📝 ОПИСАНИЕ ЗВУКА")
        print("   Опишите звук, который хотите создать.")
        print("   Примеры: 'metal clanging', 'rain on roof', 'dog barking'")
        prompt = input("🎵 Введите описание: ").strip()
        
        if prompt.lower() == 'q':
            break
        
        if not prompt:
            prompt = "The sound of metal clanging against metal"
            print(f"   Используется промт по умолчанию: {prompt}")
        
        # Длительность
        print("\n⏱️  ДЛИТЕЛЬНОСТЬ")
        duration = get_float_input(
            "   Введите длительность (сек, Enter=1.0): ",
            default=1.0,
            min_val=0.1,  # Минимум 0.1 секунды
            max_val=47.0,
            explanation="Можно вводить дробные значения: 0.5 (полсекунды), 0.3, 2.5 и т.д."
        )
        
        # Количество файлов
        print("\n📁 КОЛИЧЕСТВО ФАЙЛОВ")
        num_files = get_int_input(
            "   Сколько файлов создать? (Enter=1): ",
            default=1,
            min_val=1,
            max_val=100,
            explanation="Файлы генерируются по одному для экономии памяти"
        )
        
        # Количество шагов
        print("\n⚙️  КАЧЕСТВО ГЕНЕРАЦИИ")
        num_steps = get_int_input(
            "   Количество шагов (Enter=100): ",
            default=100,
            min_val=30,
            max_val=200,
            explanation="Больше шагов = лучше качество. Для коротких звуков можно уменьшить до 50-70"
        )
        
        # Создание папки
        output_folder = create_output_folder(prompt, str(base_output_dir))
        print(f"\n📂 Создана папка: {output_folder}")
        
        # Параметры
        additional_params = {
            "num_inference_steps": num_steps,
            "negative_prompt": "Low quality, distorted",
            "model": "stabilityai/stable-audio-open-1.0",
            "duration_seconds": duration,
        }
        
        # Сохранение информации
        save_prompt_info(output_folder, prompt, duration, num_files, additional_params)
        print("📄 Сохранен prompt_info.txt")
        
        print(f"\n🎵 Генерация: {prompt}")
        print(f"⏱️  Длительность: {duration:.2f} сек ({duration*1000:.0f} мс)")
        print(f"📁 Количество файлов: {num_files}")
        print(f"⚙️  Шагов: {num_steps}")
        print("\n⏳ Ожидайте, идет генерация...")
        
        # Генерация файлов по одному
        successful = 0
        failed = 0
        total_time = 0
        
        for i in range(1, num_files + 1):
            print(f"\n[{i}/{num_files}] Генерация...")
            
            start_time = time.time()
            
            try:
                # Уникальный seed для каждого файла
                seed = int(time.time() * 1000) + i * 1000
                
                # Генерация одного файла
                audio_waveform = generate_single_audio(
                    pipe, prompt, duration, num_steps, seed
                )
                
                elapsed = time.time() - start_time
                total_time += elapsed
                
                # Сохранение
                output_path = os.path.join(output_folder, f"variation_{i:03d}.wav")
                
                # Конвертация в numpy
                if isinstance(audio_waveform, torch.Tensor):
                    output = audio_waveform.T.float().cpu().numpy()
                else:
                    output = audio_waveform
                
                sf.write(output_path, output, pipe.vae.sampling_rate)
                
                successful += 1
                
                file_size = os.path.getsize(output_path) / 1024
                actual_duration = len(output) / pipe.vae.sampling_rate if output.ndim == 1 else output.shape[0] / pipe.vae.sampling_rate
                
                print(f"  ✅ variation_{i:03d}.wav ({file_size:.0f} KB, {actual_duration:.2f} сек, {elapsed:.1f} сек)")
                
                # Обновление информации в файле
                with open(os.path.join(output_folder, "prompt_info.txt"), "a", encoding="utf-8") as f:
                    f.write(f"\nФайл {i}: variation_{i:03d}.wav\n")
                    f.write(f"  Seed: {seed}\n")
                    f.write(f"  Запрошенная длительность: {duration:.2f} сек\n")
                    f.write(f"  Фактическая длительность: {actual_duration:.2f} сек\n")
                    f.write(f"  Время генерации: {elapsed:.1f} сек\n")
                    f.write(f"  Размер: {file_size:.0f} KB\n")
                
                # Очистка памяти
                clear_gpu_memory()
                
            except torch.cuda.OutOfMemoryError:
                failed += 1
                print(f"  ❌ Недостаточно памяти для файла {i}")
                print("  Попробуйте уменьшить длительность или количество шагов")
                clear_gpu_memory()
                break
            except Exception as e:
                failed += 1
                print(f"  ❌ Ошибка: {e}")
                clear_gpu_memory()
        
        # Итоги
        print(f"\n{'=' * 70}")
        print("✅ ГЕНЕРАЦИЯ ЗАВЕРШЕНА")
        print(f"{'=' * 70}")
        print(f"📁 Успешно: {successful}/{num_files}")
        if failed > 0:
            print(f"❌ Ошибок: {failed}")
        print(f"⏱️  Общее время: {total_time:.1f} сек ({total_time/60:.1f} мин)")
        print(f"📂 Папка: {output_folder}")
        
        # Очистка
        clear_gpu_memory()
        
        # Продолжение
        print("\n" + "=" * 70)
        continue_choice = input("Создать еще один звуковой эффект? (y/n): ").strip().lower()
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