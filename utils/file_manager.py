# -*- coding: utf-8 -*-
"""
Менеджер файлов Local Transcriber
Copyright (C) 2025 Local Transcriber Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List
import tempfile

from config.logging_config import get_logger, log_function_call

logger = get_logger('file_manager')

class FileManager:
    """Менеджер для работы с файлами."""
    
    def __init__(self, base_directory: str):
        """
        Инициализация менеджера файлов.
        
        Args:
            base_directory: Базовая директория для работы
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileManager инициализирован с базовой директорией: {self.base_directory}")
    
    @log_function_call
    def get_upload_path(self, filename: str) -> Path:
        """
        Получить полный путь для загружаемого файла.
        
        Args:
            filename: Имя файла
            
        Returns:
            Path: Полный путь к файлу
        """
        file_path = self.base_directory / filename
        logger.debug(f"Сгенерирован путь для файла {filename}: {file_path}")
        return file_path
    
    @log_function_call
    def ensure_directory_exists(self, directory: str) -> Path:
        """
        Убедиться что директория существует, создать если нет.
        
        Args:
            directory: Путь к директории
            
        Returns:
            Path: Путь к директории
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Директория {dir_path} создана или уже существует")
        return dir_path
    
    @log_function_call
    def delete_file(self, file_path: str) -> bool:
        """
        Удалить файл.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            bool: True если файл удален успешно
        """
        try:
            path = Path(file_path)
            if path.exists():
                if path.is_file():
                    path.unlink()
                    logger.info(f"Файл {file_path} удален")
                elif path.is_dir():
                    shutil.rmtree(path)
                    logger.info(f"Директория {file_path} удалена")
                return True
            else:
                logger.warning(f"Файл {file_path} не существует для удаления")
                return False
        except Exception as e:
            logger.error(f"Ошибка при удалении файла {file_path}: {str(e)}")
            return False
    
    @log_function_call
    def move_file(self, source: str, destination: str) -> bool:
        """
        Переместить файл.
        
        Args:
            source: Путь к исходному файлу
            destination: Путь к целевому файлу
            
        Returns:
            bool: True если файл перемещен успешно
        """
        try:
            source_path = Path(source)
            dest_path = Path(destination)
            
            if not source_path.exists():
                logger.error(f"Исходный файл {source} не существует")
                return False
            
            # Создаем директорию назначения если её нет
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(source_path), str(dest_path))
            logger.info(f"Файл перемещен: {source} -> {destination}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при перемещении файла {source} -> {destination}: {str(e)}")
            return False
    
    @log_function_call
    def copy_file(self, source: str, destination: str) -> bool:
        """
        Скопировать файл.
        
        Args:
            source: Путь к исходному файлу
            destination: Путь к целевому файлу
            
        Returns:
            bool: True если файл скопирован успешно
        """
        try:
            source_path = Path(source)
            dest_path = Path(destination)
            
            if not source_path.exists():
                logger.error(f"Исходный файл {source} не существует")
                return False
            
            # Создаем директорию назначения если её нет
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(str(source_path), str(dest_path))
            logger.info(f"Файл скопирован: {source} -> {destination}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при копировании файла {source} -> {destination}: {str(e)}")
            return False
    
    @log_function_call
    def get_file_size(self, file_path: str) -> Optional[int]:
        """
        Получить размер файла в байтах.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Optional[int]: Размер файла в байтах или None при ошибке
        """
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                size = path.stat().st_size
                logger.debug(f"Размер файла {file_path}: {size} байт")
                return size
            else:
                logger.warning(f"Файл {file_path} не существует")
                return None
        except Exception as e:
            logger.error(f"Ошибка при получении размера файла {file_path}: {str(e)}")
            return None
    
    @log_function_call
    def list_files(self, directory: str, extension: Optional[str] = None) -> List[Path]:
        """
        Получить список файлов в директории.
        
        Args:
            directory: Путь к директории
            extension: Расширение файлов для фильтрации (например, '.wav')
            
        Returns:
            List[Path]: Список путей к файлам
        """
        try:
            dir_path = Path(directory)
            if not dir_path.exists() or not dir_path.is_dir():
                logger.warning(f"Директория {directory} не существует")
                return []
            
            files = []
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    if extension is None or file_path.suffix.lower() == extension.lower():
                        files.append(file_path)
            
            logger.debug(f"Найдено {len(files)} файлов в {directory}")
            return files
        except Exception as e:
            logger.error(f"Ошибка при получении списка файлов в {directory}: {str(e)}")
            return []
    
    @log_function_call
    def create_temp_file(self, suffix: str = ".tmp", prefix: str = "transcriber_") -> str:
        """
        Создать временный файл.
        
        Args:
            suffix: Суффикс файла
            prefix: Префикс файла
            
        Returns:
            str: Путь к временному файлу
        """
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
            os.close(temp_fd)  # Закрываем дескриптор, файл остается
            logger.debug(f"Создан временный файл: {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Ошибка при создании временного файла: {str(e)}")
            raise
    
    @log_function_call
    def create_temp_directory(self, prefix: str = "transcriber_") -> str:
        """
        Создать временную директорию.
        
        Args:
            prefix: Префикс директории
            
        Returns:
            str: Путь к временной директории
        """
        try:
            temp_dir = tempfile.mkdtemp(prefix=prefix)
            logger.debug(f"Создана временная директория: {temp_dir}")
            return temp_dir
        except Exception as e:
            logger.error(f"Ошибка при создании временной директории: {str(e)}")
            raise
    
    @log_function_call
    def cleanup_temp_files(self, temp_paths: List[str]):
        """
        Очистить временные файлы и директории.
        
        Args:
            temp_paths: Список путей для удаления
        """
        logger.info(f"Начинается очистка {len(temp_paths)} временных файлов")
        
        for temp_path in temp_paths:
            self.delete_file(temp_path)
        
        logger.info("Очистка временных файлов завершена")
    
    def get_safe_filename(self, filename: str) -> str:
        """
        Получить безопасное имя файла.
        
        Args:
            filename: Исходное имя файла
            
        Returns:
            str: Безопасное имя файла
        """
        # Удаляем или заменяем опасные символы
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        safe_name = "".join(c if c in safe_chars else "_" for c in filename)
        
        # Ограничиваем длину
        if len(safe_name) > 100:
            name_part = safe_name[:80]
            ext_part = safe_name[-20:] if '.' in safe_name[-20:] else ""
            safe_name = name_part + ext_part
        
        logger.debug(f"Безопасное имя файла: {filename} -> {safe_name}")
        return safe_name 