from faster_whisper import WhisperModel
import torch
import os
import psutil
import time
from Components.Logger import logger, timed_operation


# Старые функции транскрипции (`transcribeAudio`, `transcribe_word_level_full`)
# были удалены и заменены единой функцией `transcribe_unified`.
# Это изменение повышает производительность, избегая повторной загрузки модели
# и двойного прохода по аудиофайлу.

@timed_operation("transcribe_unified")
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
    logger.logger.info(f"Запуск единой транскрипции аудио: {audio_path}")

    # Логирование начальных системных ресурсов
    _log_system_resources("Начало транскрипции")

    segments_legacy = []
    word_level_transcription = {"segments": []}

    try:
        # Создаем прогресс-бар для транскрипции
        progress_bar = logger.create_progress_bar(
            total=100,  # Процентное представление
            desc="Транскрипция аудио",
            unit="%"
        )

        start_time = time.time()

        # Запуск транскрипции с параметрами
        logger.logger.info("Запуск модели faster-whisper для транскрипции...")
        segments_gen, info = model.transcribe(
            audio=audio_path,
            beam_size=5,
            language="ru",
            condition_on_previous_text=True,
            vad_filter=True,
            word_timestamps=True  # Ключевой параметр для получения слов
        )

        # Получаем информацию о длительности аудио
        audio_duration = getattr(info, 'duration', 0.0)
        logger.logger.info(f"Длительность аудио: {audio_duration:.2f} секунд")

        processed_segments = 0
        total_segments = 0

        # Предварительный подсчет общего количества сегментов для прогресс-бара
        segments_list = list(segments_gen)
        total_segments = len(segments_list)
        segments_gen = iter(segments_list)  # Сбрасываем генератор

        logger.logger.info(f"Обнаружено {total_segments} сегментов для обработки")

        for i, seg in enumerate(segments_gen):
            segment_start = time.time()

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

            processed_segments += 1

            # Обновляем прогресс-бар
            progress = int((processed_segments / total_segments) * 100) if total_segments > 0 else 100
            progress_bar.update(max(0, progress - progress_bar.n))
            progress_bar.set_postfix({
                "Сегмент": f"{processed_segments}/{total_segments}",
                "Время": f"{seg.start:.1f}s-{seg.end:.1f}s",
                "Слова": len(words_arr)
            })

            # Логирование обработки сегмента
            segment_time = time.time() - segment_start
            logger.logger.debug(f"Обработан сегмент {processed_segments}/{total_segments}: "
                              f"время={segment_time:.3f}s, слова={len(words_arr)}, "
                              f"текст='{seg.text[:50]}...'")

            # Периодическое логирование системных ресурсов
            if processed_segments % 10 == 0 or processed_segments == total_segments:
                _log_system_resources(f"Обработка сегмента {processed_segments}/{total_segments}")

        progress_bar.close()

        total_time = time.time() - start_time
        logger.logger.info(f"Транскрипция завершена за {total_time:.2f} секунд")
        logger.logger.info(f"Обработано {len(segments_legacy)} сегментов, "
                          f"скорость: {audio_duration/total_time:.2f}x")

        # Финальное логирование ресурсов
        _log_system_resources("Завершение транскрипции")

        return segments_legacy, word_level_transcription

    except Exception as e:
        logger.logger.error(f"Ошибка в единой функции транскрипции: {e}")
        import traceback
        traceback.print_exc()
        # Возвращаем пустые структуры в случае ошибки
        return [], {"segments": []}


def _log_system_resources(context: str = ""):
    """
    Логирует текущие системные ресурсы (CPU, память, GPU если доступен).

    Args:
        context (str): Контекст для логирования (например, "Начало транскрипции")
    """
    try:
        # CPU и память
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used / (1024**3)  # GB
        memory_total = memory.total / (1024**3)  # GB

        resource_info = {
            "cpu_percent": f"{cpu_percent:.1f}%",
            "memory_percent": f"{memory_percent:.1f}%",
            "memory_used": f"{memory_used:.2f}GB",
            "memory_total": f"{memory_total:.2f}GB"
        }

        # GPU ресурсы если доступны
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                resource_info["gpu_count"] = gpu_count

                for i in range(gpu_count):
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                    gpu_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                    gpu_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB

                    resource_info[f"gpu_{i}_memory"] = f"{gpu_allocated:.2f}/{gpu_reserved:.2f}/{gpu_memory:.2f}GB"

                    # Температура GPU если доступна
                    try:
                        import subprocess
                        result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu',
                                               '--format=csv,noheader,nounits'],
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            temp = result.stdout.strip().split('\n')[i]
                            resource_info[f"gpu_{i}_temp"] = f"{temp}°C"
                    except:
                        pass
        except Exception as e:
            logger.logger.debug(f"Не удалось получить GPU информацию: {e}")

        context_str = f" [{context}]" if context else ""
        logger.logger.info(f"Системные ресурсы{context_str}: {resource_info}")

    except Exception as e:
        logger.logger.warning(f"Ошибка при логировании системных ресурсов: {e}")


# Блок if __name__ == "__main__" был удален, так как он
# использовал устаревшие функции для тестирования.
