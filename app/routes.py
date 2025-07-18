# -*- coding: utf-8 -*-
"""
Маршруты Flask приложения Local Transcriber
Copyright (C) 2025 Local Transcriber Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import os
import threading
from pathlib import Path
from flask import Blueprint, render_template, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from config.logging_config import get_logger
from app.models import job_manager, UploadedFile, FileType, ProcessingJob
from utils.file_manager import FileManager
from utils.audio_processor import AudioProcessor

logger = get_logger('routes')
main = Blueprint('main', __name__)

def allowed_file(filename):
    """
    Проверить, разрешен ли тип файла.
    
    Args:
        filename: Имя файла
        
    Returns:
        bool: True если файл разрешен
    """
    return '.' in filename and \
           Path(filename).suffix.lower() in current_app.config['ALLOWED_EXTENSIONS']

def get_file_type(filename):
    """
    Определить тип файла.
    
    Args:
        filename: Имя файла
        
    Returns:
        FileType: Тип файла
    """
    extension = Path(filename).suffix.lower()
    type_mapping = {
        '.wav': FileType.WAV,
        '.mp3': FileType.MP3,
        '.mp4': FileType.MP4,
        '.mkv': FileType.MKV
    }
    return type_mapping.get(extension, FileType.WAV)

@main.route('/')
def index():
    """Главная страница."""
    logger.info("Запрос главной страницы")
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_files():
    """
    Загрузка файлов.
    
    Returns:
        JSON: Ответ с информацией о загруженных файлах
    """
    logger.info("Получен запрос на загрузку файлов")
    
    try:
        # Проверка наличия файлов
        if 'files' not in request.files:
            logger.warning("Нет файлов в запросе")
            return jsonify({'error': 'Нет файлов для загрузки'}), 400
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            logger.warning("Пустые файлы в запросе")
            return jsonify({'error': 'Нет выбранных файлов'}), 400
        
        # Создание новой задачи
        job = job_manager.create_job()
        job.start_processing()
        
        file_manager = FileManager(current_app.config['UPLOAD_FOLDER'])
        uploaded_files = []
        
        # Обработка каждого файла
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                logger.info(f"Обработка файла: {file.filename}")
                
                # Безопасное имя файла
                original_filename = file.filename
                secure_name = secure_filename(original_filename)
                
                # Генерация уникального имени
                file_id = f"{job.id}_{len(uploaded_files)}_{secure_name}"
                file_path = file_manager.get_upload_path(file_id)
                
                # Сохранение файла
                file.save(file_path)
                file_size = os.path.getsize(file_path)
                
                # Создание объекта файла
                uploaded_file = UploadedFile(
                    filename=file_id,
                    original_filename=original_filename,
                    file_type=get_file_type(original_filename),
                    size_bytes=file_size,
                    file_path=str(file_path)
                )
                
                job.add_file(uploaded_file)
                uploaded_files.append(uploaded_file)
                
                logger.info(f"Файл {original_filename} загружен успешно ({file_size} байт)")
            else:
                logger.warning(f"Файл {file.filename} не разрешен")
                return jsonify({'error': f'Файл {file.filename} не поддерживается'}), 400
        
        # Обновление прогресса
        job.update_progress(job.status, "Файлы загружены", 10.0)
        
        # Запуск обработки в отдельном потоке
        threading.Thread(
            target=process_audio_async,
            args=(job.id, current_app.config['UPLOAD_FOLDER'], current_app.config['RESULTS_FOLDER']),
            daemon=True
        ).start()
        
        logger.info(f"Загружено {len(uploaded_files)} файлов для задачи {job.id}")
        return jsonify({
            'job_id': job.id,
            'files_count': len(uploaded_files),
            'total_size': job.get_total_size(),
            'message': 'Файлы загружены, начинается обработка'
        })
    
    except RequestEntityTooLarge:
        logger.error("Файл слишком большой")
        return jsonify({'error': 'Файл слишком большой (максимум 2GB)'}), 413
    except Exception as e:
        logger.error(f"Ошибка при загрузке файлов: {str(e)}", exc_info=True)
        return jsonify({'error': f'Ошибка загрузки: {str(e)}'}), 500

@main.route('/status/<job_id>')
def get_job_status(job_id):
    """
    Получить статус задачи.
    
    Args:
        job_id: ID задачи
        
    Returns:
        JSON: Статус задачи
    """
    logger.debug(f"Запрос статуса задачи: {job_id}")
    
    job = job_manager.get_job(job_id)
    if not job:
        logger.warning(f"Задача {job_id} не найдена")
        return jsonify({'error': 'Задача не найдена'}), 404
    
    return jsonify(job.to_dict())

@main.route('/download/<job_id>')
def download_result(job_id):
    """
    Скачать результат обработки.
    
    Args:
        job_id: ID задачи
        
    Returns:
        File: Файл с результатом
    """
    logger.info(f"Запрос на скачивание результата задачи: {job_id}")
    
    job = job_manager.get_job(job_id)
    if not job:
        logger.warning(f"Задача {job_id} не найдена")
        return jsonify({'error': 'Задача не найдена'}), 404
    
    if not job.result_file_path or not os.path.exists(job.result_file_path):
        logger.warning(f"Результат задачи {job_id} не готов или не существует")
        return jsonify({'error': 'Результат не готов'}), 404
    
    logger.info(f"Отправка файла результата: {job.result_file_path}")
    return send_file(
        job.result_file_path,
        as_attachment=True,
        download_name=f"transcription_{job_id}.txt",
        mimetype='text/plain'
    )

@main.route('/jobs')
def list_jobs():
    """
    Список всех задач.
    
    Returns:
        JSON: Список задач
    """
    logger.debug("Запрос списка задач")
    
    jobs_data = []
    for job in job_manager.jobs.values():
        jobs_data.append(job.to_dict())
    
    return jsonify({
        'jobs': jobs_data,
        'total': len(jobs_data)
    })

@main.route('/delete/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """
    Удалить задачу.
    
    Args:
        job_id: ID задачи
        
    Returns:
        JSON: Результат удаления
    """
    logger.info(f"Запрос на удаление задачи: {job_id}")
    
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Задача не найдена'}), 404
    
    # Удаление файлов
    file_manager = FileManager(current_app.config['UPLOAD_FOLDER'])
    for file in job.files:
        file_manager.delete_file(file.file_path)
    
    if job.result_file_path:
        file_manager.delete_file(job.result_file_path)
    
    if job.combined_audio_path:
        file_manager.delete_file(job.combined_audio_path)
    
    # Удаление задачи
    job_manager.delete_job(job_id)
    
    return jsonify({'message': 'Задача удалена'})

@main.errorhandler(404)
def not_found(error):
    """Обработчик 404 ошибки."""
    logger.warning(f"404 ошибка: {request.url}")
    return jsonify({'error': 'Страница не найдена'}), 404

@main.errorhandler(500)
def internal_error(error):
    """Обработчик 500 ошибки."""
    logger.error(f"500 ошибка: {str(error)}", exc_info=True)
    return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

def process_audio_async(job_id, upload_folder, results_folder):
    """
    Асинхронная обработка аудио.
    
    Args:
        job_id: ID задачи
        upload_folder: Папка для загруженных файлов
        results_folder: Папка для результатов
    """
    logger.info(f"Начало асинхронной обработки задачи: {job_id}")
    
    job = job_manager.get_job(job_id)
    if not job:
        logger.error(f"Задача {job_id} не найдена для обработки")
        return
    
    try:
        from utils.transcription import TranscriptionProcessor
        
        # Инициализация процессора
        processor = TranscriptionProcessor(
            upload_folder=upload_folder,
            results_folder=results_folder
        )
        
        # Запуск полной обработки
        result_path = processor.process_full_pipeline(job)
        
        if result_path:
            job.complete_processing(result_path)
            logger.info(f"Задача {job_id} обработана успешно")
        else:
            job.set_error("Ошибка в процессе обработки")
            logger.error(f"Не удалось обработать задачу {job_id}")
    
    except Exception as e:
        error_msg = f"Ошибка при обработке: {str(e)}"
        job.set_error(error_msg)
        logger.error(f"Ошибка в асинхронной обработке задачи {job_id}: {error_msg}", exc_info=True) 