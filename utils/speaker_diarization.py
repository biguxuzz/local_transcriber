# -*- coding: utf-8 -*-
"""
Диаризация спикеров Local Transcriber
Copyright (C) 2025 Local Transcriber Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import os
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import numpy as np

from config.logging_config import get_logger, log_function_call, log_processing_step, log_progress
from app.models import SpeakerSegment, ProcessingJob, ProcessingStatus

logger = get_logger('speaker_diarization')

class SpeakerDiarizer:
    """Диаризатор спикеров с использованием pyannote."""
    
    def __init__(self, model_name: str = "pyannote/speaker-diarization", 
                 hf_token: Optional[str] = None, min_duration: float = 1.0):
        """
        Инициализация диаризатора.
        
        Args:
            model_name: Название модели для диаризации
            hf_token: Токен Hugging Face для доступа к модели
            min_duration: Минимальная длительность сегмента спикера в секундах
        """
        self.model_name = model_name
        self.hf_token = hf_token
        self.min_duration = min_duration
        self.pipeline = None
        
        logger.info(f"SpeakerDiarizer инициализирован: model={model_name}, min_duration={min_duration}")
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Инициализация пайплайна диаризации с ONNX Runtime."""
        try:
            log_processing_step("Инициализация модели диаризации", self.model_name)
            
            # ИСПРАВЛЕНИЕ для Windows: настройка стратегии загрузки файлов ПЕРЕД импортом
            import platform
            if platform.system() == 'Windows':
                logger.info("Windows обнаружена - настройка стратегии копирования файлов для SpeechBrain")
                os.environ['SPEECHBRAIN_CACHE_DIR'] = os.path.expanduser('~/.cache/speechbrain')
                # Принудительно отключаем символические ссылки
                try:
                    import speechbrain.utils.fetching as sb_fetch
                    # Устанавливаем переменные окружения для принудительного копирования
                    os.environ['SB_USE_SYMLINKS'] = 'False'
                    os.environ['SPEECHBRAIN_CACHE_LOCAL_STRATEGY'] = 'copy'
                    # Пытаемся патчить функцию создания ссылок  
                    if hasattr(sb_fetch, 'link_with_strategy'):
                        original_link = sb_fetch.link_with_strategy
                        def patched_link(src, dst, strategy=None):
                            # Принудительно используем копирование вместо ссылок
                            import shutil
                            import pathlib
                            src_path = pathlib.Path(src)
                            dst_path = pathlib.Path(dst)
                            dst_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_path, dst_path)
                            return dst_path
                        sb_fetch.link_with_strategy = patched_link
                    logger.info("SpeechBrain патчен для использования копирования вместо символических ссылок")
                except Exception as e:
                    logger.warning(f"Не удалось патчить speechbrain: {str(e)}")
            
            # Проверяем доступность pyannote.audio
            try:
                from pyannote.audio import Pipeline
            except ImportError as e:
                logger.error(f"pyannote.audio не установлен: {str(e)}")
                raise RuntimeError("pyannote.audio не установлен. Установите: pip install pyannote.audio")
            
            # Настройка ONNX Runtime для CPU оптимизации
            self._setup_onnx_runtime()
            
            # Загружаем пайплайн с обработкой ошибок
            try:
                if self.hf_token:
                    self.pipeline = Pipeline.from_pretrained(
                        self.model_name, 
                        use_auth_token=self.hf_token
                    )
                else:
                    self.pipeline = Pipeline.from_pretrained(self.model_name)
            except Exception as e:
                error_msg = str(e).lower()
                if "gated" in error_msg or "private" in error_msg or "403" in error_msg:
                    logger.warning("Модель требует принятия условий использования на Hugging Face")
                    logger.info("Попробуйте посетить https://huggingface.co/pyannote/speaker-diarization")
                    logger.info("и принять условия использования, затем повторите попытку")
                    raise RuntimeError(f"Модель {self.model_name} требует принятия условий использования на Hugging Face")
                else:
                    raise
            
            # Принудительно используем CPU с ONNX оптимизацией
            self._configure_for_cpu()
            
            logger.info("Пайплайн диаризации успешно инициализирован с ONNX Runtime")
        
        except Exception as e:
            logger.error(f"Ошибка инициализации пайплайна диаризации: {str(e)}")
            raise RuntimeError(f"Не удалось загрузить модель диаризации: {str(e)}")
    
    def _setup_onnx_runtime(self):
        """Настройка ONNX Runtime для оптимизации на CPU."""
        try:
            import onnxruntime
            
            # Настройка провайдеров для ONNX Runtime
            providers = ['CPUExecutionProvider']
            
            # Проверяем доступные провайдеры
            available_providers = onnxruntime.get_available_providers()
            logger.info(f"Доступные ONNX провайдеры: {available_providers}")
            
            # Устанавливаем CPU провайдер
            os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
            os.environ['ONNXRUNTIME_PROVIDERS'] = 'CPUExecutionProvider'
            
            # ИСПРАВЛЕНИЕ для Windows: используем копирование вместо символических ссылок
            import platform
            if platform.system() == 'Windows':
                logger.info("Windows обнаружена - настройка SpeechBrain для копирования файлов")
                os.environ['SPEECHBRAIN_LOCAL_STRATEGY'] = 'copy'
            
            logger.info("ONNX Runtime настроен для CPU оптимизации")
            
        except ImportError:
            logger.warning("ONNX Runtime не найден, используется стандартный PyTorch")
        except Exception as e:
            logger.warning(f"Ошибка настройки ONNX Runtime: {str(e)}")
    
    def _configure_for_cpu(self):
        """Принудительная настройка для работы на CPU."""
        try:
            import torch
            
            # Принудительно отключаем CUDA
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            # Устанавливаем модель на CPU
            if hasattr(self.pipeline, 'to'):
                self.pipeline = self.pipeline.to(torch.device("cpu"))
            
            # Оптимизация для CPU
            torch.set_num_threads(os.cpu_count())
            
            logger.info("Модель диаризации настроена для работы на CPU")
            
        except ImportError:
            logger.info("PyTorch не найден, используется CPU по умолчанию")
        except Exception as e:
            logger.warning(f"Ошибка настройки CPU: {str(e)}")
    
    @log_function_call
    def diarize_audio(self, audio_path: str) -> List[SpeakerSegment]:
        """
        Выполнить диаризацию аудио файла.
        
        Args:
            audio_path: Путь к аудио файлу
            
        Returns:
            List[SpeakerSegment]: Список сегментов спикеров
        """
        try:
            log_processing_step("Диаризация аудио", Path(audio_path).name)
            
            if not self.pipeline:
                raise RuntimeError("Пайплайн диаризации не инициализирован")
            
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")
            
            # Выполняем диаризацию
            logger.info(f"Запуск диаризации для файла: {audio_path}")
            diarization = self.pipeline(audio_path)
            
            # Конвертируем результаты в наш формат
            segments = []
            speaker_mapping = {}
            speaker_counter = 1
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Создаем единообразные имена спикеров
                if speaker not in speaker_mapping:
                    speaker_mapping[speaker] = f"SPEAKER_{speaker_counter:02d}"
                    speaker_counter += 1
                
                # Проверяем минимальную длительность
                duration = turn.end - turn.start
                if duration >= self.min_duration:
                    segment = SpeakerSegment(
                        speaker_id=speaker_mapping[speaker],
                        start_time=turn.start,
                        end_time=turn.end
                    )
                    segments.append(segment)
                    
                    logger.debug(f"Добавлен сегмент: {segment.speaker_id} "
                               f"({segment.start_time:.2f}s - {segment.end_time:.2f}s)")
                else:
                    logger.debug(f"Пропущен короткий сегмент спикера {speaker}: {duration:.2f}s")
            
            # Сортируем сегменты по времени
            segments.sort(key=lambda x: x.start_time)
            
            unique_speakers = len(speaker_mapping)
            logger.info(f"Диаризация завершена: найдено {len(segments)} сегментов от {unique_speakers} спикеров")
            
            # Выводим статистику
            self._log_diarization_stats(segments, speaker_mapping)
            
            return segments
        
        except Exception as e:
            logger.error(f"Ошибка диаризации аудио файла {audio_path}: {str(e)}")
            raise
    
    def _log_diarization_stats(self, segments: List[SpeakerSegment], speaker_mapping: Dict[str, str]):
        """Логирование статистики диаризации."""
        try:
            # Общая статистика
            total_duration = sum(segment.duration for segment in segments)
            logger.info(f"Общая длительность сегментов: {total_duration:.2f}s")
            
            # Статистика по спикерам
            speaker_stats = {}
            for segment in segments:
                if segment.speaker_id not in speaker_stats:
                    speaker_stats[segment.speaker_id] = {
                        'segments': 0,
                        'total_duration': 0.0,
                        'avg_segment_duration': 0.0
                    }
                
                speaker_stats[segment.speaker_id]['segments'] += 1
                speaker_stats[segment.speaker_id]['total_duration'] += segment.duration
            
            # Вычисляем средние значения
            for speaker_id, stats in speaker_stats.items():
                stats['avg_segment_duration'] = stats['total_duration'] / stats['segments']
                
                logger.info(f"Спикер {speaker_id}: {stats['segments']} сегментов, "
                           f"общая длительность: {stats['total_duration']:.2f}s, "
                           f"средняя длительность сегмента: {stats['avg_segment_duration']:.2f}s")
        
        except Exception as e:
            logger.warning(f"Ошибка при выводе статистики диаризации: {str(e)}")
    
    @log_function_call
    def merge_close_segments(self, segments: List[SpeakerSegment], 
                           max_gap: float = 1.0) -> List[SpeakerSegment]:
        """
        Объединить близкие сегменты одного спикера.
        
        Args:
            segments: Список сегментов
            max_gap: Максимальный промежуток между сегментами для объединения (сек)
            
        Returns:
            List[SpeakerSegment]: Объединенные сегменты
        """
        if not segments:
            return segments
        
        logger.info(f"Объединение близких сегментов (max_gap={max_gap}s)")
        
        merged_segments = []
        current_segment = segments[0]
        
        for next_segment in segments[1:]:
            # Проверяем, можно ли объединить с текущим сегментом
            if (next_segment.speaker_id == current_segment.speaker_id and
                next_segment.start_time - current_segment.end_time <= max_gap):
                
                # Объединяем сегменты
                current_segment = SpeakerSegment(
                    speaker_id=current_segment.speaker_id,
                    start_time=current_segment.start_time,
                    end_time=next_segment.end_time
                )
                
                logger.debug(f"Объединены сегменты спикера {current_segment.speaker_id}")
            else:
                # Добавляем текущий сегмент и начинаем новый
                merged_segments.append(current_segment)
                current_segment = next_segment
        
        # Добавляем последний сегмент
        merged_segments.append(current_segment)
        
        original_count = len(segments)
        merged_count = len(merged_segments)
        logger.info(f"Объединение завершено: {original_count} -> {merged_count} сегментов")
        
        return merged_segments
    
    @log_function_call
    def filter_short_segments(self, segments: List[SpeakerSegment], 
                            min_duration: Optional[float] = None) -> List[SpeakerSegment]:
        """
        Отфильтровать короткие сегменты.
        
        Args:
            segments: Список сегментов
            min_duration: Минимальная длительность (если не указано, используется self.min_duration)
            
        Returns:
            List[SpeakerSegment]: Отфильтрованные сегменты
        """
        if min_duration is None:
            min_duration = self.min_duration
        
        logger.info(f"Фильтрация коротких сегментов (min_duration={min_duration}s)")
        
        filtered_segments = [
            segment for segment in segments 
            if segment.duration >= min_duration
        ]
        
        original_count = len(segments)
        filtered_count = len(filtered_segments)
        removed_count = original_count - filtered_count
        
        if removed_count > 0:
            logger.info(f"Удалено {removed_count} коротких сегментов")
        
        return filtered_segments
    
    @log_function_call
    def process_audio_with_diarization(self, job: ProcessingJob, audio_path: str) -> List[SpeakerSegment]:
        """
        Обработать аудио файл с диаризацией спикеров.
        
        Args:
            job: Задача обработки
            audio_path: Путь к аудио файлу
            
        Returns:
            List[SpeakerSegment]: Список сегментов спикеров
        """
        try:
            log_processing_step("Обработка аудио с диаризацией", job.id)
            
            # Выполняем диаризацию
            segments = self.diarize_audio(audio_path)
            
            # Объединяем близкие сегменты
            segments = self.merge_close_segments(segments)
            
            # Фильтруем короткие сегменты
            segments = self.filter_short_segments(segments)
            
            logger.info(f"Обработка завершена: {len(segments)} сегментов")
            return segments
        
        except Exception as e:
            logger.error(f"Ошибка обработки аудио с диаризацией: {str(e)}")
            raise
    
    @log_function_call
    def get_speaker_segments_for_transcription(self, segments: List[SpeakerSegment], 
                                             chunk_size: float = 30.0) -> List[Tuple[float, float, str]]:
        """
        Получить сегменты для транскрипции с учетом диаризации.
        
        Args:
            segments: Список сегментов спикеров
            chunk_size: Размер чанка в секундах
            
        Returns:
            List[Tuple[float, float, str]]: Список (начало, конец, спикер)
        """
        if not segments:
            return []
        
        logger.info(f"Подготовка сегментов для транскрипции (chunk_size={chunk_size}s)")
        
        transcription_segments = []
        
        for segment in segments:
            # Разбиваем длинные сегменты на чанки
            start_time = segment.start_time
            end_time = segment.end_time
            
            while start_time < end_time:
                chunk_end = min(start_time + chunk_size, end_time)
                
                transcription_segments.append((
                    start_time,
                    chunk_end,
                    segment.speaker_id
                ))
                
                start_time = chunk_end
        
        logger.info(f"Подготовлено {len(transcription_segments)} сегментов для транскрипции")
        return transcription_segments
    
    def cleanup(self):
        """Очистка ресурсов."""
        logger.info("Очистка ресурсов диаризатора")
        if self.pipeline:
            try:
                # Освобождаем GPU память если используется
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            self.pipeline = None 