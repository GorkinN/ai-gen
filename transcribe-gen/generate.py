# transcribe-gen/generate.py
import sys
import os
import shutil
import wave
from dataclasses import replace
from pathlib import Path

root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import time
from datetime import datetime, timedelta

from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
    load_audio_item,
    resolve_device,
)

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"


def create_output_folder(audio_path, base_dir):
    """Создает папку для транскрипции"""
    stem = Path(audio_path).stem
    safe_name = "".join(
        c for c in stem[:50]
        if c.isalnum() or c in " -_"
    ).strip().replace(" ", "_")

    if not safe_name:
        safe_name = "unnamed_audio"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(base_dir, f"{timestamp}_{safe_name}")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def format_srt_time(seconds):
    """Форматирует секунды в таймкод SRT"""
    total_ms = max(0, int(round(float(seconds) * 1000)))
    td = timedelta(milliseconds=total_ms)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    hours += td.days * 24
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{td.microseconds // 1000:03d}"


def format_clock(seconds):
    """Форматирует секунды как mm:ss"""
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def write_wav(path, samples, sample_rate):
    """Сохраняет моно PCM 16-bit WAV"""
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def split_audio_chunks(samples, sample_rate, chunk_seconds, overlap_seconds):
    """Режет аудио на overlapping-куски. Возвращает список (start_sec, end_sec, audio)."""
    total = len(samples)
    if total == 0:
        return []

    chunk_len = max(1, int(round(chunk_seconds * sample_rate)))
    overlap_len = int(round(overlap_seconds * sample_rate))
    overlap_len = max(0, min(overlap_len, chunk_len - 1))
    step = max(1, chunk_len - overlap_len)

    ranges = []
    start = 0
    while start < total:
        end = min(start + chunk_len, total)
        ranges.append((start, end))
        if end >= total:
            break
        start += step

    min_keep = max(int(sample_rate * 1.0), overlap_len)
    if len(ranges) >= 2:
        last_start, last_end = ranges[-1]
        if last_end - last_start < min_keep:
            prev_start, _ = ranges[-2]
            ranges[-2] = (prev_start, last_end)
            ranges.pop()

    return [
        (start / sample_rate, end / sample_rate, samples[start:end])
        for start, end in ranges
    ]


def offset_segments(segments, offset_seconds, min_relative_start=0.0):
    """Сдвигает таймкоды куска на абсолютное время и отбрасывает дубли в overlap."""
    shifted = []
    for segment in segments:
        if segment.start < min_relative_start:
            continue
        shifted.append(
            replace(
                segment,
                start=segment.start + offset_seconds,
                end=segment.end + offset_seconds,
            )
        )
    return shifted


def save_results(folder_path, audio_path, result_text, segments, elapsed, additional_params=None):
    """Сохраняет сырой текст, сегменты, SRT и метаданные"""
    raw_path = os.path.join(folder_path, "transcript_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(result_text or "")

    readable_path = os.path.join(folder_path, "transcript.txt")
    with open(readable_path, "w", encoding="utf-8") as f:
        if segments:
            for segment in segments:
                f.write(
                    f"[{segment.start:.2f} - {segment.end:.2f}] "
                    f"{segment.speaker}: {segment.text}\n"
                )
        else:
            f.write(result_text or "")

    srt_path = os.path.join(folder_path, "transcript.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}\n")
            speaker = segment.speaker or "S??"
            text = (segment.text or "").strip()
            f.write(f"[{speaker}] {text}\n\n")

    info_path = os.path.join(folder_path, "prompt_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("MOSS-Transcribe-Diarize - Информация о транскрипции\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Аудио: {audio_path}\n")
        f.write(f"Сегментов: {len(segments)}\n")
        f.write(f"Время обработки: {elapsed:.1f} сек\n")

        if additional_params:
            f.write("\nДополнительные параметры:\n")
            for key, value in additional_params.items():
                f.write(f"  {key}: {value}\n")

        f.write("\n" + "=" * 60 + "\n")
        if torch.cuda.is_available():
            f.write(f"Устройство: GPU ({torch.cuda.get_device_name(0)})\n")
            f.write(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
        else:
            f.write("Устройство: CPU\n")


def get_int_input(prompt_text, default=2048, min_val=256, max_val=65536, explanation=""):
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


def get_float_input(prompt_text, default=60.0, min_val=5.0, max_val=600.0, explanation=""):
    """Запрашивает число с плавающей точкой"""
    if explanation:
        print(f"  💡 {explanation}")

    while True:
        try:
            user_input = input(prompt_text).strip()
            if not user_input:
                return default

            value = float(user_input.replace(",", "."))

            if min_val <= value <= max_val:
                return value
            else:
                print(f"  ⚠️  Значение должно быть от {min_val} до {max_val}")
        except ValueError:
            print("  ⚠️  Введите число!")


def normalize_audio_path(raw_path):
    """Убирает кавычки и проверяет, что файл существует"""
    audio_path = raw_path.strip().strip('"').strip("'")
    if not audio_path:
        return None, "Путь не указан"

    path = Path(audio_path)
    if not path.exists():
        return None, f"Файл не найден: {audio_path}"
    if not path.is_file():
        return None, f"Это не файл: {audio_path}"
    return str(path.resolve()), None


def copy_source_audio(audio_path, output_folder):
    """Копирует исходный файл в папку результата"""
    src = Path(audio_path)
    dest = Path(output_folder) / f"source{src.suffix.lower() or '.wav'}"
    shutil.copy2(src, dest)
    return dest


def transcribe_file(model, processor, audio_path, max_new_tokens, device, dtype):
    """Транскрибирует один аудиофайл"""
    messages = build_transcription_messages(audio_path)
    result = generate_transcription(
        model,
        processor,
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        device=device,
        dtype=dtype,
    )
    result_text = result.get("text", "") if isinstance(result, dict) else str(result)
    return result_text, list(parse_transcript(result_text))


def main():
    print("=" * 70)
    print("📝 MOSS-Transcribe-Diarize - Транскрипция аудио")
    print("=" * 70)

    device = resolve_device("auto")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if device.type == "cuda":
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
    else:
        print("⚠️ CUDA не найдена, используется CPU")

    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ Устройство: {device}")

    print("\n📥 Загрузка модели MOSS-Transcribe-Diarize...")
    print("   Это может занять 1-2 минуты")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            dtype="auto",
        ).to(dtype=dtype).to(device).eval()

        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )
        print("✓ Модель загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return

    sample_rate = int(processor.feature_extractor.sampling_rate)
    base_output_dir = Path(__file__).parent / "transcribe_output"
    base_output_dir.mkdir(exist_ok=True)

    while True:
        print("\n" + "=" * 70)
        print("ТРАНСКРИПЦИЯ АУДИО")
        print("=" * 70)

        print("\n📁 АУДИОФАЙЛ")
        print("   Введите путь без кавычек. Для выхода введите 'q'.")
        audio_input = input("🎵 Путь: ").strip()

        if audio_input.lower() == "q":
            break

        audio_path, error = normalize_audio_path(audio_input)
        if error:
            print(f"  ❌ {error}")
            continue

        print("\n✂️  НАРЕЗКА")
        chunk_seconds = get_float_input(
            "   Длина куска в секундах (Enter=60): ",
            default=60.0,
            min_val=10.0,
            max_val=300.0,
            explanation="Модель часто обрывает длинный файл. Короткие куски распознаются целиком.",
        )
        overlap_seconds = get_float_input(
            "   Перекрытие кусков в секундах (Enter=5): ",
            default=5.0,
            min_val=0.0,
            max_val=30.0,
            explanation="Перекрытие убирает обрыв фразы на границе кусков",
        )
        if overlap_seconds >= chunk_seconds:
            overlap_seconds = max(0.0, chunk_seconds / 5)
            print(f"   ⚠️ Перекрытие уменьшено до {overlap_seconds:.1f} сек")

        print("\n⚙️  ДЛИНА ГЕНЕРАЦИИ")
        max_new_tokens = get_int_input(
            "   max_new_tokens на кусок (Enter=2048): ",
            default=2048,
            min_val=256,
            max_val=65536,
            explanation="Лимит токенов на один кусок, не на весь файл",
        )

        output_folder = create_output_folder(audio_path, str(base_output_dir))
        chunks_dir = Path(output_folder) / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        print(f"\n📂 Создана папка: {output_folder}")

        try:
            copied_path = copy_source_audio(audio_path, output_folder)
            print(f"💾 Копия исходника: {copied_path.name}")

            print("🔄 Загрузка и нарезка аудио...")
            samples = load_audio_item(str(copied_path), sampling_rate=sample_rate)
            duration = len(samples) / sample_rate
            chunks = split_audio_chunks(samples, sample_rate, chunk_seconds, overlap_seconds)
            print(f"✓ Длительность: {duration:.1f} сек, кусков: {len(chunks)}")
        except Exception as e:
            print(f"❌ Не удалось подготовить аудио: {e}")
            continue

        additional_params = {
            "model": MODEL_ID,
            "max_new_tokens": max_new_tokens,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "chunks": len(chunks),
            "duration_seconds": round(duration, 2),
            "sample_rate": sample_rate,
            "do_sample": False,
            "dtype": str(dtype),
        }

        print(f"\n📝 Транскрипция: {audio_path}")
        print(f"✂️ Куски: {len(chunks)} × {chunk_seconds:.0f} сек, overlap {overlap_seconds:.0f} сек")
        print(f"⚙️ max_new_tokens: {max_new_tokens}")
        print("\n⏳ Ожидайте, куски обрабатываются по очереди...")

        start_time = time.time()
        all_segments = []
        raw_parts = []

        try:
            for i, (chunk_start, chunk_end, chunk_audio) in enumerate(chunks, start=1):
                chunk_name = f"chunk_{i:03d}_{format_clock(chunk_start).replace(':', '-')}_{format_clock(chunk_end).replace(':', '-')}.wav"
                chunk_path = chunks_dir / chunk_name
                write_wav(chunk_path, chunk_audio, sample_rate)

                print(
                    f"\n[{i}/{len(chunks)}] Кусок {format_clock(chunk_start)}-{format_clock(chunk_end)} "
                    f"({chunk_end - chunk_start:.1f} сек)"
                )

                chunk_t0 = time.time()
                result_text, segments = transcribe_file(
                    model,
                    processor,
                    str(chunk_path),
                    max_new_tokens,
                    device,
                    dtype,
                )
                chunk_elapsed = time.time() - chunk_t0

                min_relative_start = (overlap_seconds / 2) if i > 1 else 0.0
                shifted = offset_segments(segments, chunk_start, min_relative_start=min_relative_start)
                all_segments.extend(shifted)
                raw_parts.append(
                    f"===== {chunk_name} | {format_clock(chunk_start)}-{format_clock(chunk_end)} =====\n"
                    f"{result_text}\n"
                )

                chunk_raw_path = chunks_dir / f"{chunk_path.stem}.txt"
                with open(chunk_raw_path, "w", encoding="utf-8") as f:
                    f.write(result_text or "")

                print(f"  ✅ сегментов: {len(shifted)} ({chunk_elapsed:.1f} сек)")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_segments.sort(key=lambda item: (item.start, item.end))
            elapsed = time.time() - start_time
            combined_raw = "\n".join(raw_parts)

            save_results(
                output_folder,
                audio_path,
                combined_raw,
                all_segments,
                elapsed,
                additional_params,
            )

            print(f"\n{'=' * 70}")
            print("✅ ГОТОВО")
            print(f"{'=' * 70}")
            print(f"⏱️ Время: {elapsed:.1f} сек")
            print(f"✂️ Кусков: {len(chunks)}")
            print(f"📁 Сегментов: {len(all_segments)}")
            print(f"📂 Папка: {output_folder}")

            print("\n--- Транскрипт ---")
            if all_segments:
                preview = all_segments[:20]
                for segment in preview:
                    print(
                        f"[{segment.start:.2f}-{segment.end:.2f}] "
                        f"{segment.speaker}: {segment.text}"
                    )
                if len(all_segments) > 20:
                    print(f"... еще {len(all_segments) - 20} сегментов в transcript.txt")
            else:
                print(combined_raw)

        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cont = input("\nТранскрибировать еще? (y/n): ").strip().lower()
        if cont != "y":
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
