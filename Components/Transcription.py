from faster_whisper import WhisperModel
import torch
import os


def _select_runtime():
    """
    Dynamically selects the runtime environment (CPU or GPU) for Whisper.
    """
    try:
        # Check for CUDA availability
        has_cuda = torch.cuda.is_available()
    except Exception:
        # Fallback if torch or CUDA check fails
        has_cuda = False

    if has_cuda:
        # GPU (CUDA) environment
        device = "cuda"
        compute_type = "float16"  # Faster on modern GPUs
        print("CUDA is available. Using GPU with float16.")
    else:
        # CPU environment
        device = "cpu"
        compute_type = "int8"     # Optimized for CPU
        print("CUDA not available. Using CPU with int8.")

    # Standard model for both environments
    model_size = "large-v3"
    
    # Set CPU threads for CTranslate2
    cpu_threads = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
    
    return model_size, device, compute_type, cpu_threads


# Старые функции транскрипции (`transcribeAudio`, `transcribe_word_level_full`)
# были удалены и заменены единой функцией `transcribe_unified`.
# Это изменение повышает производительность, избегая повторной загрузки модели
# и двойного прохода по аудиофайлу.

def transcribe_unified(audio_path, model):
    """
    Выполняет транскрипцию аудио одним вызовом и возвращает две структуры данных:
    сегменты и полную транскрипцию с разбивкой по словам.

    Это единая точка входа для транскрипции, заменяющая двойные вызовы
    transcribeAudio и transcribe_word_level_full.

    Args:
        audio_path (str): Путь к аудиофайлу.
        model (WhisperModel): Загруженная модель faster-whisper.

    Returns:
        tuple[list, dict]: Кортеж, содержащий:
        - list: Список сегментов в формате [[text, start, end], ...].
        - dict: Полная транскрипция на уровне слов в формате
                {'segments': [{'text': str, 'start': float, 'end': float, 'words': [...]}, ...]}.
    """
    print("Запуск единой транскрипции (сегменты и слова)...")
    segments_legacy = []
    word_level_transcription = {"segments": []}

    try:
        segments_gen, info = model.transcribe(
            audio=audio_path,
            beam_size=5,
            language="ru",
            condition_on_previous_text=True,
            vad_filter=True,
            word_timestamps=True  # Ключевой параметр для получения слов
        )

        for seg in segments_gen:
            # 1. Формируем структуру для транскрипции на уровне слов
            seg_dict = {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text
            }
            words_arr = []
            if getattr(seg, "words", None):
                for w in seg.words:
                    words_arr.append({
                        "word": w.word,
                        "start": float(w.start),
                        "end": float(w.end)
                    })
            seg_dict["words"] = words_arr
            word_level_transcription["segments"].append(seg_dict)

            # 2. Формируем "старую" структуру сегментов
            segments_legacy.append([seg.text, float(seg.start), float(seg.end)])

        print(f"Единая транскрипция завершена. Найдено {len(segments_legacy)} сегментов.")
        return segments_legacy, word_level_transcription

    except Exception as e:
        print(f"Ошибка в единой функции транскрипции: {e}")
        import traceback
        traceback.print_exc()
        # Возвращаем пустые структуры в случае ошибки
        return [], {"segments": []}

# Блок if __name__ == "__main__" был удален, так как он
# использовал устаревшие функции для тестирования.
