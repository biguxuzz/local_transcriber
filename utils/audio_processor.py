# -*- coding: utf-8 -*-
"""
Процессор аудио файлов Local Transcriber
Copyright (C) 2025 Local Transcriber Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import ffmpeg

from config.logging_config import get_logger, log_function_call, log_processing_step, log_progress
from app.models import UploadedFile, ProcessingJob, ProcessingStatus
from utils.file_manager import FileManager

logger = get_logger('audio_processor')

class AudioProcessor:
    """Процессор для обработки аудио файлов."""
    
    def __init__(self, upload_folder: str, sample_rate: int = 16000, channels: int = 1):
        """
        Инициализация процессора аудио.
        
        Args:
            upload_folder: Папка для загрузок
            sample_rate: Частота дискретизации (по умолчанию 16kHz)
            channels: Количество каналов (по умолчанию 1 - моно)
        """
        self.upload_folder = Path(upload_folder)
        self.sample_rate = sample_rate
        self.channels = channels
        self.file_manager = FileManager(upload_folder)
        
        logger.info(f"AudioProcessor инициализирован: sample_rate={sample_rate}, channels={channels}")
        self._check_ffmpeg_availability()
    
    def _check_ffmpeg_availability(self):
        """Проверить доступность ffmpeg."""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                    capture_output=True, 
                                    text=True, 
                                    timeout=10)
            if result.returncode == 0:
                logger.info("FFmpeg доступен")
            else:
                logger.warning("FFmpeg недоступен")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"Ошибка проверки FFmpeg: {str(e)}")
            raise RuntimeError("FFmpeg не найден. Установите FFmpeg для работы с аудио.")
    
    @log_function_call
    def get_audio_info(self, file_path: str) -> Optional[dict]:
        """
        Получить информацию об аудио файле.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            dict: Информация о файле (длительность, частота дискретизации, каналы)
        """
        try:
            probe = ffmpeg.probe(file_path)
            audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
            
            if not audio_streams:
                logger.error(f"Аудио поток не найден в файле {file_path}")
                return None
            
            audio_stream = audio_streams[0]
            
            info = {
                'duration': float(probe['format'].get('duration', 0)),
                'sample_rate': int(audio_stream.get('sample_rate', 0)),
                'channels': int(audio_stream.get('channels', 0)),
                'codec': audio_stream.get('codec_name', 'unknown'),
                'bitrate': int(probe['format'].get('bit_rate', 0)),
                'format': probe['format'].get('format_name', 'unknown')
            }
            
            logger.debug(f"Информация о файле {file_path}: {info}")
            return info
        except Exception as e:
            logger.error(f"Ошибка получения информации о файле {file_path}: {str(e)}")
            return None
    
    @log_function_call
    def convert_to_wav(self, input_path: str, output_path: str) -> bool:
        """
        Конвертировать аудио файл в WAV формат.
        
        Args:
            input_path: Путь к исходному файлу
            output_path: Путь к выходному файлу
            
        Returns:
            bool: True если конвертация успешна
        """
        try:
            log_processing_step("Конвертация в WAV", f"Файл: {Path(input_path).name}")
            
            # Конвертация с помощью ffmpeg
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(
                stream,
                output_path,
                ar=self.sample_rate,  # частота дискретизации
                ac=self.channels,     # количество каналов
                acodec='pcm_s16le',   # кодек
                f='wav'               # формат
            )
            
            # Запуск с перезаписью файла
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            # Проверка результата
            if Path(output_path).exists():
                file_size = self.file_manager.get_file_size(output_path)
                logger.info(f"Конвертация завершена: {input_path} -> {output_path} ({file_size} байт)")
                return True
            else:
                logger.error(f"Файл {output_path} не создан после конвертации")
                return False
        
        except Exception as e:
            logger.error(f"Ошибка конвертации {input_path}: {str(e)}")
            return False
    
    @log_function_call
    def combine_audio_files(self, file_paths: List[str], output_path: str) -> bool:
        """
        Объединить несколько аудио файлов в один.
        
        Args:
            file_paths: Список путей к файлам для объединения
            output_path: Путь к выходному файлу
            
        Returns:
            bool: True если объединение успешно
        """
        try:
            if len(file_paths) == 1:
                # Если файл один, просто копируем
                return self.file_manager.copy_file(file_paths[0], output_path)
            
            log_processing_step("Объединение аудио файлов", f"Файлов: {len(file_paths)}")
            
            # Создаем входные потоки для всех файлов
            input_streams = [ffmpeg.input(file_path) for file_path in file_paths]
            
            # Объединяем потоки
            output_stream = ffmpeg.concat(*input_streams, v=0, a=1)
            
            # Настройка выходного потока
            output_stream = ffmpeg.output(
                output_stream,
                output_path,
                ar=self.sample_rate,
                ac=self.channels,
                acodec='pcm_s16le',
                f='wav'
            )
            
            # Запуск объединения
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            
            # Проверка результата
            if Path(output_path).exists():
                file_size = self.file_manager.get_file_size(output_path)
                logger.info(f"Объединение завершено: {len(file_paths)} файлов -> {output_path} ({file_size} байт)")
                return True
            else:
                logger.error(f"Файл {output_path} не создан после объединения")
                return False
        
        except Exception as e:
            logger.error(f"Ошибка объединения файлов: {str(e)}")
            return False
    
    @log_function_call
    def process_uploaded_files(self, job: ProcessingJob) -> Optional[str]:
        """
        Обработать загруженные файлы (конвертация и объединение).
        
        Args:
            job: Задача обработки
            
        Returns:
            str: Путь к итоговому WAV файлу или None при ошибке
        """
        try:
            log_processing_step("Обработка загруженных файлов", f"Файлов: {len(job.files)}")
            
            if not job.files:
                logger.error("Нет файлов для обработки")
                return None
            
            # Обновляем статус
            job.update_progress(ProcessingStatus.CONVERTING, "Конвертация файлов", 20.0)
            
            converted_files = []
            temp_files = []
            
            # Конвертируем каждый файл в WAV
            for i, file in enumerate(job.files):
                log_progress(i + 1, len(job.files), "Конвертация файлов")
                
                # Путь к исходному файлу
                input_path = file.file_path
                
                # Если файл уже WAV, проверяем его параметры
                if file.file_type.value == 'wav':
                    audio_info = self.get_audio_info(input_path)
                    if (audio_info and 
                        audio_info['sample_rate'] == self.sample_rate and 
                        audio_info['channels'] == self.channels):
                        # Файл уже в нужном формате
                        converted_files.append(input_path)
                        file.duration_seconds = audio_info['duration']
                        logger.info(f"Файл {file.original_filename} уже в нужном формате")
                        continue
                
                # Создаем путь для конвертированного файла
                converted_filename = f"{job.id}_{i}_converted.wav"
                converted_path = str(self.upload_folder / converted_filename)
                temp_files.append(converted_path)
                
                # Конвертируем файл
                if self.convert_to_wav(input_path, converted_path):
                    converted_files.append(converted_path)
                    
                    # Получаем информацию о конвертированном файле
                    audio_info = self.get_audio_info(converted_path)
                    if audio_info:
                        file.duration_seconds = audio_info['duration']
                else:
                    logger.error(f"Не удалось конвертировать файл {file.original_filename}")
                    # Очистка временных файлов при ошибке
                    self.file_manager.cleanup_temp_files(temp_files)
                    return None
            
            # Обновляем статус
            job.update_progress(ProcessingStatus.COMBINING, "Объединение файлов", 40.0)
            
            # Объединяем конвертированные файлы
            if len(converted_files) > 1:
                combined_filename = f"{job.id}_combined.wav"
                combined_path = str(self.upload_folder / combined_filename)
                
                if self.combine_audio_files(converted_files, combined_path):
                    # Получаем информацию об объединенном файле
                    audio_info = self.get_audio_info(combined_path)
                    if audio_info:
                        job.total_duration = audio_info['duration']
                    
                    # Очищаем временные файлы конвертации
                    self.file_manager.cleanup_temp_files(temp_files)
                    
                    logger.info(f"Файлы успешно обработаны и объединены: {combined_path}")
                    return combined_path
                else:
                    logger.error("Не удалось объединить файлы")
                    self.file_manager.cleanup_temp_files(temp_files)
                    return None
            else:
                # Только один файл
                final_path = converted_files[0]
                audio_info = self.get_audio_info(final_path)
                if audio_info:
                    job.total_duration = audio_info['duration']
                
                logger.info(f"Обработан единственный файл: {final_path}")
                return final_path
        
        except Exception as e:
            logger.error(f"Ошибка обработки файлов: {str(e)}")
            return None
    
    @log_function_call
    def extract_audio_segment(self, input_path: str, output_path: str, 
                             start_time: float, duration: float) -> bool:
        """
        Извлечь сегмент аудио.
        
        Args:
            input_path: Путь к исходному файлу
            output_path: Путь к выходному файлу
            start_time: Время начала в секундах
            duration: Длительность в секундах
            
        Returns:
            bool: True если извлечение успешно
        """
        try:
            stream = ffmpeg.input(input_path, ss=start_time, t=duration)
            stream = ffmpeg.output(
                stream,
                output_path,
                ar=self.sample_rate,
                ac=self.channels,
                acodec='pcm_s16le',
                f='wav'
            )
            
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            if Path(output_path).exists():
                logger.debug(f"Извлечен сегмент: {start_time}s-{start_time + duration}s -> {output_path}")
                return True
            else:
                logger.error(f"Не удалось создать сегмент {output_path}")
                return False
        
        except Exception as e:
            logger.error(f"Ошибка извлечения сегмента: {str(e)}")
            return False
    
    def cleanup_processed_files(self, job: ProcessingJob):
        """
        Очистить обработанные файлы.
        
        Args:
            job: Задача обработки
        """
        logger.info(f"Очистка файлов для задачи {job.id}")
        
        # Удаляем исходные загруженные файлы
        for file in job.files:
            self.file_manager.delete_file(file.file_path)
        
        # Удаляем объединенный файл если он есть
        if job.combined_audio_path:
            self.file_manager.delete_file(job.combined_audio_path) 