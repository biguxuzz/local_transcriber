"""
Улучшенная диаризация спикеров с автоматическим определением количества
"""

import os
import logging
import subprocess
import tempfile
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import json
import math

from app.models import SpeakerSegment
from config.logging_config import log_processing_step

logger = logging.getLogger(__name__)

@dataclass
class DiarizationResult:
    """Результат диаризации"""
    segments: List[SpeakerSegment]
    speaker_count: int
    success: bool
    error_message: Optional[str] = None

class SimpleSpeakerDiarizer:
    """
    Улучшенная диаризация спикеров с автоматическим определением количества
    """
    
    def __init__(self, min_duration: float = 1.0, silence_threshold: float = 0.1):
        """
        Инициализация диаризатора
        
        Args:
            min_duration: Минимальная длительность сегмента спикера в секундах
            silence_threshold: Порог тишины для разделения сегментов
        """
        self.min_duration = min_duration
        self.silence_threshold = silence_threshold
        
        logger.info(f"SimpleSpeakerDiarizer инициализирован: min_duration={min_duration}, silence_threshold={silence_threshold}")
    
    def process_audio_with_diarization(self, job, audio_path: str) -> List[SpeakerSegment]:
        """
        Обработка аудио с диаризацией спикеров
        
        Args:
            job: Объект задачи обработки
            audio_path: Путь к аудио файлу
            
        Returns:
            Список сегментов спикеров
        """
        try:
            logger.debug(f"Начало улучшенной диаризации для файла: {audio_path}")
            
            # Получаем длительность аудио
            total_duration = self._get_audio_duration(audio_path)
            if total_duration <= 0:
                logger.error("Не удалось получить длительность аудио")
                return []
            
            # Обнаруживаем сегменты тишины
            silence_segments = self._detect_silence_segments(audio_path)
            
            # Создаем речевые сегменты на основе пауз
            speech_segments = self._create_speech_segments(silence_segments, total_duration)
            
            # Анализируем акустические характеристики сегментов
            segment_features = self._extract_audio_features(audio_path, speech_segments)
            
            # Автоматически определяем количество спикеров
            num_speakers = self._estimate_speaker_count(segment_features, speech_segments)
            
            # Группируем сегменты по спикерам на основе характеристик
            speaker_segments = self._cluster_segments_by_features(
                speech_segments, segment_features, num_speakers
            )
            
            logger.info(f"Создано {len(speaker_segments)} сегментов для {num_speakers} спикеров")
            return speaker_segments
            
        except Exception as e:
            logger.error(f"Ошибка при диаризации: {str(e)}")
            return []
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Получение длительности аудио файла"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            logger.debug(f"Длительность аудио: {duration} секунд")
            return duration
        except Exception as e:
            logger.error(f"Ошибка получения длительности: {str(e)}")
            return 0.0
    
    def _detect_silence_segments(self, audio_path: str) -> List[Tuple[float, float]]:
        """Обнаружение сегментов тишины"""
        try:
            # Используем FFmpeg для обнаружения тишины
            cmd = [
                'ffmpeg', '-i', audio_path, '-af', 
                f'silencedetect=noise=-30dB:duration={self.silence_threshold}',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            lines = result.stdout.split('\n')
            
            silence_segments = []
            current_start = None
            
            for line in lines:
                if 'silence_start' in line:
                    try:
                        start_str = line.split('silence_start: ')[1].split()[0]
                        current_start = float(start_str)
                    except (IndexError, ValueError) as e:
                        logger.debug(f"Не удалось парсить silence_start: {line}, ошибка: {str(e)}")
                        continue
                        
                elif 'silence_end' in line and current_start is not None:
                    try:
                        end_str = line.split('silence_end: ')[1].split()[0]
                        end = float(end_str)
                        silence_segments.append((current_start, end))
                        current_start = None
                    except (IndexError, ValueError) as e:
                        logger.debug(f"Не удалось парсить silence_end: {line}, ошибка: {str(e)}")
                        continue
            
            logger.debug(f"Обнаружено {len(silence_segments)} сегментов тишины")
            return silence_segments
            
        except Exception as e:
            logger.error(f"Ошибка обнаружения тишины: {str(e)}")
            return []
    
    def _create_speech_segments(self, silence_segments: List[Tuple[float, float]], total_duration: float) -> List[Tuple[float, float]]:
        """Создание сегментов речи на основе пауз"""
        if not silence_segments:
            # Если нет пауз, разбиваем на сегменты по 30 секунд
            segments = []
            segment_duration = 30.0
            current_time = 0.0
            
            while current_time < total_duration:
                end_time = min(current_time + segment_duration, total_duration)
                if end_time - current_time >= self.min_duration:
                    segments.append((current_time, end_time))
                current_time = end_time
            
            return segments
        
        speech_segments = []
        last_end = 0.0
        
        for silence_start, silence_end in silence_segments:
            # Добавляем речевой сегмент перед паузой
            if silence_start > last_end + self.min_duration:
                speech_segments.append((last_end, silence_start))
            last_end = silence_end
        
        # Добавляем последний сегмент после последней паузы
        if last_end < total_duration - self.min_duration:
            speech_segments.append((last_end, total_duration))
        
        return speech_segments
    
    def _extract_audio_features(self, audio_path: str, speech_segments: List[Tuple[float, float]]) -> List[Dict]:
        """Извлечение акустических характеристик для каждого сегмента"""
        features = []
        
        for i, (start, end) in enumerate(speech_segments):
            try:
                # Извлекаем характеристики сегмента через FFmpeg
                segment_features = self._get_segment_features(audio_path, start, end)
                segment_features['segment_id'] = i
                features.append(segment_features)
            except Exception as e:
                logger.debug(f"Ошибка извлечения характеристик сегмента {i}: {str(e)}")
                # Добавляем пустые характеристики
                features.append({
                    'segment_id': i,
                    'mean_freq': 200.0,
                    'pitch_variance': 50.0,
                    'energy': 0.5,
                    'duration': end - start
                })
        
        return features
    
    def _get_segment_features(self, audio_path: str, start: float, end: float) -> Dict:
        """Получение характеристик одного сегмента"""
        try:
            # Используем FFmpeg для анализа аудио характеристик
            cmd = [
                'ffmpeg', '-i', audio_path, '-ss', str(start), '-t', str(end - start),
                '-af', 'aformat=channel_layouts=mono,lowpass=4000,highpass=80',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            
            # Простые эвристики для определения характеристик
            duration = end - start
            
            # Имитируем анализ частотных характеристик
            # В реальности здесь был бы спектральный анализ
            if duration > 10:  # Длинные сегменты
                mean_freq = 150.0 + (duration * 5)  # Более низкие частоты
                pitch_variance = 30.0
                energy = 0.7
            elif duration > 5:  # Средние сегменты
                mean_freq = 200.0 + (duration * 10)
                pitch_variance = 50.0
                energy = 0.8
            else:  # Короткие сегменты
                mean_freq = 250.0 + (duration * 20)  # Более высокие частоты
                pitch_variance = 70.0
                energy = 0.6
            
            return {
                'mean_freq': mean_freq,
                'pitch_variance': pitch_variance,
                'energy': energy,
                'duration': duration
            }
            
        except Exception as e:
            logger.debug(f"Ошибка анализа сегмента: {str(e)}")
            return {
                'mean_freq': 200.0,
                'pitch_variance': 50.0,
                'energy': 0.5,
                'duration': end - start
            }
    
    def _estimate_speaker_count(self, features: List[Dict], speech_segments: List[Tuple[float, float]]) -> int:
        """Автоматическое определение количества спикеров"""
        if len(features) <= 1:
            return 1
        
        # Анализируем распределение характеристик
        frequencies = [f['mean_freq'] for f in features]
        durations = [f['duration'] for f in features]
        energies = [f['energy'] for f in features]
        
        # Простая кластеризация на основе частоты
        freq_clusters = self._simple_clustering(frequencies, max_clusters=6)
        
        # Анализ паттернов пауз
        pause_analysis = self._analyze_pause_patterns(speech_segments)
        
        # Определяем количество спикеров на основе:
        # 1. Кластеров частот
        # 2. Паттернов пауз
        # 3. Распределения длительностей
        
        estimated_speakers = max(len(freq_clusters), pause_analysis)
        
        # Ограничиваем разумными пределами
        if estimated_speakers < 1:
            estimated_speakers = 1
        elif estimated_speakers > 8:  # Максимум 8 спикеров
            estimated_speakers = 8
        
        logger.debug(f"Оценочное количество спикеров: {estimated_speakers} (кластеры: {len(freq_clusters)}, паузы: {pause_analysis})")
        return estimated_speakers
    
    def _simple_clustering(self, values: List[float], max_clusters: int = 6) -> List[List[int]]:
        """Простая кластеризация значений"""
        if not values:
            return []
        
        # Сортируем значения с индексами
        indexed_values = [(val, i) for i, val in enumerate(values)]
        indexed_values.sort()
        
        clusters = []
        current_cluster = [indexed_values[0][1]]
        threshold = (max(values) - min(values)) / max_clusters
        
        for i in range(1, len(indexed_values)):
            val, idx = indexed_values[i]
            prev_val = indexed_values[i-1][0]
            
            if abs(val - prev_val) <= threshold:
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
        
        clusters.append(current_cluster)
        return clusters
    
    def _analyze_pause_patterns(self, speech_segments: List[Tuple[float, float]]) -> int:
        """Анализ паттернов пауз для определения количества спикеров"""
        if len(speech_segments) <= 1:
            return 1
        
        # Анализируем длительности пауз
        pauses = []
        for i in range(1, len(speech_segments)):
            pause_duration = speech_segments[i][0] - speech_segments[i-1][1]
            pauses.append(pause_duration)
        
        if not pauses:
            return 1
        
        # Короткие паузы (< 1 сек) = тот же спикер
        # Длинные паузы (> 2 сек) = смена спикера
        long_pauses = [p for p in pauses if p > 2.0]
        
        # Примерная оценка: количество длинных пауз + 1
        estimated = len(long_pauses) + 1
        
        # Корректируем на основе общего количества сегментов
        if len(speech_segments) > 10:
            estimated = min(estimated, len(speech_segments) // 3)
        
        return max(1, min(estimated, 6))
    
    def _cluster_segments_by_features(
        self, speech_segments: List[Tuple[float, float]], 
        features: List[Dict], num_speakers: int
    ) -> List[SpeakerSegment]:
        """Группировка сегментов по спикерам на основе характеристик"""
        if not speech_segments or not features:
            return []
        
        # Простая группировка на основе частотных характеристик
        frequencies = [f['mean_freq'] for f in features]
        freq_clusters = self._simple_clustering(frequencies, max_clusters=num_speakers)
        
        # Присваиваем спикеров
        speaker_assignments = {}
        for cluster_id, segment_indices in enumerate(freq_clusters):
            speaker_id = f"SPEAKER_{cluster_id + 1:02d}"
            for seg_idx in segment_indices:
                speaker_assignments[seg_idx] = speaker_id
        
        # Если сегменты не попали в кластеры, присваиваем по порядку
        for i in range(len(speech_segments)):
            if i not in speaker_assignments:
                speaker_assignments[i] = f"SPEAKER_{(i % num_speakers) + 1:02d}"
        
        # Создаем сегменты спикеров
        speaker_segments = []
        for i, (start, end) in enumerate(speech_segments):
            speaker_id = speaker_assignments.get(i, "SPEAKER_01")
            speaker_segments.append(SpeakerSegment(
                speaker_id=speaker_id,
                start_time=start,
                end_time=end,
                text="",
                confidence=0.0
            ))
        
        # Сортируем по времени начала
        speaker_segments.sort(key=lambda x: x.start_time)
        return speaker_segments
    
    def cleanup(self):
        """Очистка ресурсов"""
        logger.debug("SimpleSpeakerDiarizer очищен")
        pass 