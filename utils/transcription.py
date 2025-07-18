# -*- coding: utf-8 -*-
"""
Транскрибация аудио Local Transcriber
Copyright (C) 2025 Local Transcriber Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import tempfile

from config.logging_config import get_logger, log_function_call, log_processing_step, log_progress
from app.models import ProcessingJob, ProcessingStatus, SpeakerSegment
from utils.file_manager import FileManager
from utils.audio_processor import AudioProcessor
from utils.speaker_diarization import SpeakerDiarizer

logger = get_logger('transcription')

class TranscriptionProcessor:
    """Основной процессор транскрибации, объединяющий все этапы."""
    
    def __init__(self, upload_folder: str, results_folder: str, 
                 whisper_model: str = "base", hf_token: Optional[str] = None):
        """
        Инициализация процессора транскрибации.
        
        Args:
            upload_folder: Папка для загруженных файлов
            results_folder: Папка для результатов
            whisper_model: Модель Whisper для транскрибации
            hf_token: Токен Hugging Face
        """
        self.upload_folder = Path(upload_folder)
        self.results_folder = Path(results_folder)
        self.whisper_model_name = whisper_model
        self.hf_token = hf_token
        self.whisper_model = None
        
        # Создание директорий
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        
        # Инициализация компонентов
        self.file_manager = FileManager(str(upload_folder))
        self.audio_processor = AudioProcessor(str(upload_folder))
        
        # Инициализация диаризатора (опционально)
        self.speaker_diarizer = None
        logger.info("Попытка инициализации pyannote диаризатора...")
        try:
            # Пробуем только pyannote диаризацию
            from .speaker_diarization import SpeakerDiarizer
            logger.info("Модуль speaker_diarization импортирован успешно")
            self.speaker_diarizer = SpeakerDiarizer(hf_token=hf_token)
            logger.info("✅ Pyannote диаризатор спикеров инициализирован успешно!")
        except Exception as e:
            logger.error(f"❌ Не удалось инициализировать pyannote диаризатор: {str(e)}")
            logger.error(f"Тип ошибки: {type(e).__name__}")
            import traceback
            logger.error(f"Полная ошибка:\n{traceback.format_exc()}")
            logger.info("Приложение будет работать без диаризации спикеров - транскрибация целых файлов")
        
        logger.info(f"TranscriptionProcessor инициализирован: model={whisper_model}")
        self._initialize_whisper()
    
    def _initialize_whisper(self):
        """Инициализация модели Whisper."""
        try:
            log_processing_step("Инициализация модели Whisper", self.whisper_model_name)
            
            import whisper
            
            # Загружаем модель Whisper
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            
            # Проверяем доступность GPU
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("Whisper будет использовать GPU")
                else:
                    logger.info("Whisper будет использовать CPU")
            except ImportError:
                logger.info("PyTorch не найден, Whisper будет использовать CPU")
            
            logger.info("Модель Whisper успешно загружена")
        
        except Exception as e:
            logger.error(f"Ошибка загрузки модели Whisper: {str(e)}")
            raise RuntimeError(f"Не удалось загрузить модель Whisper: {str(e)}")
    
    @log_function_call
    def transcribe_audio_segment(self, audio_path: str, start_time: float, 
                               end_time: float) -> Optional[str]:
        """
        Транскрибировать сегмент аудио.
        
        Args:
            audio_path: Путь к аудио файлу
            start_time: Время начала сегмента в секундах
            end_time: Время конца сегмента в секундах
            
        Returns:
            str: Транскрибированный текст или None при ошибке
        """
        try:
            if not self.whisper_model:
                raise RuntimeError("Модель Whisper не инициализирована")
            
            # Создаем временный файл для сегмента
            temp_file = self.file_manager.create_temp_file(suffix=".wav")
            
            try:
                # Извлекаем сегмент
                if not self.audio_processor.extract_audio_segment(
                    audio_path, temp_file, start_time, end_time - start_time):
                    logger.error(f"Не удалось извлечь сегмент {start_time}-{end_time}")
                    return None
                
                # Транскрибируем сегмент
                result = self.whisper_model.transcribe(
                    temp_file,
                    language='ru',  # Русский язык по умолчанию
                    task='transcribe'
                )
                
                text = result.get('text', '').strip()
                
                if text:
                    logger.debug(f"Транскрибирован сегмент {start_time:.2f}-{end_time:.2f}s: '{text[:50]}...'")
                else:
                    logger.debug(f"Пустая транскрибация для сегмента {start_time:.2f}-{end_time:.2f}s")
                
                return text
            
            finally:
                # Удаляем временный файл
                self.file_manager.delete_file(temp_file)
        
        except Exception as e:
            logger.error(f"Ошибка транскрибации сегмента {start_time}-{end_time}: {str(e)}")
            return None
    
    @log_function_call
    def transcribe_full_audio(self, audio_path: str) -> Optional[str]:
        """
        Транскрибировать полный аудио файл без диаризации.
        
        Args:
            audio_path: Путь к аудио файлу
            
        Returns:
            str: Полный транскрибированный текст
        """
        try:
            if not self.whisper_model:
                raise RuntimeError("Модель Whisper не инициализирована")
            
            logger.info(f"Транскрибация полного файла: {audio_path}")
            
            result = self.whisper_model.transcribe(
                audio_path,
                language='ru',
                task='transcribe',
                verbose=False
            )
            
            text = result.get('text', '').strip()
            logger.info(f"Полная транскрибация завершена, длина текста: {len(text)} символов")
            
            return text
        
        except Exception as e:
            logger.error(f"Ошибка полной транскрибации {audio_path}: {str(e)}")
            return None
    
    @log_function_call
    def transcribe_with_speakers(self, job: ProcessingJob, audio_path: str, 
                               segments: List[SpeakerSegment]) -> bool:
        """
        Транскрибировать аудио с привязкой к спикерам.
        
        Args:
            job: Задача обработки
            audio_path: Путь к аудио файлу
            segments: Сегменты спикеров
            
        Returns:
            bool: True если транскрибация успешна
        """
        try:
            job.update_progress(ProcessingStatus.TRANSCRIBING, "Транскрибация сегментов", 70.0)
            
            if not segments:
                logger.warning("Нет сегментов для транскрибации")
                return False
            
            total_segments = len(segments)
            transcribed_segments = []
            
            # Транскрибируем каждый сегмент
            for i, segment in enumerate(segments):
                log_progress(i + 1, total_segments, "Транскрибация сегментов")
                
                # Транскрибируем сегмент
                text = self.transcribe_audio_segment(
                    audio_path, 
                    segment.start_time, 
                    segment.end_time
                )
                
                if text:
                    # Обновляем сегмент с текстом
                    segment.text = text
                    transcribed_segments.append(segment)
                    logger.debug(f"Сегмент {segment.speaker_id} транскрибирован: '{text[:50]}...'")
                else:
                    logger.warning(f"Пустая транскрибация для сегмента {segment.speaker_id}")
                
                # Обновляем прогресс
                progress = 70.0 + (i + 1) / total_segments * 20.0
                job.update_progress(ProcessingStatus.TRANSCRIBING, 
                                  f"Транскрибация {i+1}/{total_segments}", progress)
            
            # Обновляем сегменты в задаче
            job.speaker_segments = transcribed_segments
            
            successful_count = len([s for s in transcribed_segments if s.text.strip()])
            logger.info(f"Транскрибация завершена: {successful_count}/{total_segments} "
                       f"сегментов с текстом")
            
            return successful_count > 0
        
        except Exception as e:
            logger.error(f"Ошибка транскрибации с спикерами: {str(e)}")
            return False
    
    @log_function_call
    def format_transcription_result(self, segments: List[SpeakerSegment]) -> str:
        """
        Форматировать результат транскрибации.
        
        Args:
            segments: Сегменты с транскрибированным текстом
            
        Returns:
            str: Отформатированный текст
        """
        if not segments:
            return "Транскрибация не содержит текста."
        
        lines = []
        lines.append("# Результат транскрибации")
        lines.append(f"# Создано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"# Количество сегментов: {len(segments)}")
        
        unique_speakers = len(set(s.speaker_id for s in segments if s.text.strip()))
        lines.append(f"# Количество спикеров: {unique_speakers}")
        lines.append("")
        
        # Добавляем сегменты
        for segment in segments:
            if not segment.text.strip():
                continue
            
            # Форматируем время
            time_str = segment.formatted_time
            
            # Добавляем строку транскрибации
            line = f"{time_str} [{segment.speaker_id}] - {segment.text.strip()}"
            lines.append(line)
        
        # Добавляем статистику в конец
        lines.append("")
        lines.append("# Статистика")
        
        # Статистика по спикерам
        speaker_stats = {}
        for segment in segments:
            if not segment.text.strip():
                continue
            
            if segment.speaker_id not in speaker_stats:
                speaker_stats[segment.speaker_id] = {
                    'segments': 0,
                    'total_duration': 0.0,
                    'word_count': 0
                }
            
            speaker_stats[segment.speaker_id]['segments'] += 1
            speaker_stats[segment.speaker_id]['total_duration'] += segment.duration
            speaker_stats[segment.speaker_id]['word_count'] += len(segment.text.split())
        
        for speaker_id, stats in speaker_stats.items():
            lines.append(f"# {speaker_id}: {stats['segments']} сегментов, "
                        f"{stats['total_duration']:.1f}с, {stats['word_count']} слов")
        
        result = "\n".join(lines)
        logger.info(f"Результат отформатирован: {len(lines)} строк")
        
        return result
    
    @log_function_call
    def save_transcription_result(self, job: ProcessingJob, formatted_text: str) -> str:
        """
        Сохранить результат транскрибации в файл.
        
        Args:
            job: Задача обработки
            formatted_text: Отформатированный текст
            
        Returns:
            str: Путь к файлу результата
        """
        try:
            # Создаем имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcription_{job.id}_{timestamp}.txt"
            file_path = self.results_folder / filename
            
            # Сохраняем файл
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            
            file_size = self.file_manager.get_file_size(str(file_path))
            logger.info(f"Результат сохранен: {file_path} ({file_size} байт)")
            
            return str(file_path)
        
        except Exception as e:
            logger.error(f"Ошибка сохранения результата: {str(e)}")
            raise
    
    @log_function_call
    def process_full_pipeline(self, job: ProcessingJob) -> Optional[str]:
        """
        Выполнить полный пайплайн обработки.
        
        Args:
            job: Задача обработки
            
        Returns:
            str: Путь к файлу результата или None при ошибке
        """
        temp_files = []
        
        try:
            log_processing_step("Начало полного пайплайна", f"Задача {job.id}")
            
            # Этап 1: Обработка аудио файлов
            combined_audio_path = self.audio_processor.process_uploaded_files(job)
            if not combined_audio_path:
                job.set_error("Ошибка обработки аудио файлов")
                return None
            
            job.combined_audio_path = combined_audio_path
            
            # Этап 2: Диаризация спикеров (если доступна)
            segments = []
            if self.speaker_diarizer:
                try:
                    segments = self.speaker_diarizer.process_audio_with_diarization(job, combined_audio_path)
                    if segments:
                        # Этап 3: Транскрибация с привязкой к спикерам
                        if not self.transcribe_with_speakers(job, combined_audio_path, segments):
                            job.set_error("Ошибка транскрибации")
                            return None
                    else:
                        logger.warning("Диаризация не обнаружила спикеров")
                except Exception as e:
                    logger.warning(f"Ошибка диаризации: {str(e)}")
            
            # Если диаризация недоступна или не удалась, выполняем простую транскрибацию
            if not segments:
                logger.info("Выполняется простая транскрибация без диаризации")
                job.update_progress(ProcessingStatus.TRANSCRIBING, "Транскрибация без диаризации", 70.0)
                full_text = self.transcribe_full_audio(combined_audio_path)
                
                if full_text:
                    # Создаем единственный сегмент
                    segments = [SpeakerSegment(
                        speaker_id="SPEAKER_01",
                        start_time=0.0,
                        end_time=job.total_duration or 0.0,
                        text=full_text
                    )]
                    job.speaker_segments = segments
                    job.speaker_count = 1
                else:
                    job.set_error("Не удалось выполнить транскрибацию")
                    return None
            
            # Этап 4: Форматирование и сохранение результата
            job.update_progress(ProcessingStatus.TRANSCRIBING, "Сохранение результата", 95.0)
            
            formatted_text = self.format_transcription_result(job.speaker_segments)
            result_path = self.save_transcription_result(job, formatted_text)
            
            logger.info(f"Полный пайплайн завершен успешно для задачи {job.id}")
            return result_path
        
        except Exception as e:
            error_msg = f"Ошибка в полном пайплайне: {str(e)}"
            logger.error(error_msg)
            job.set_error(error_msg)
            return None
        
        finally:
            # Очистка временных файлов
            if temp_files:
                self.file_manager.cleanup_temp_files(temp_files)
    
    def cleanup(self):
        """Очистка ресурсов."""
        logger.info("Очистка ресурсов транскрибатора")
        
        # Очистка компонентов
        if hasattr(self, 'speaker_diarizer') and self.speaker_diarizer:
            self.speaker_diarizer.cleanup()
        
        # Очистка модели Whisper
        if self.whisper_model:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            self.whisper_model = None
        
        logger.info("Очистка ресурсов завершена") 