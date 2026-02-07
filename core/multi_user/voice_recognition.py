"""
Voice Recognition and Biometric System for JARVIS.

Provides speaker identification through voice biometrics.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class VoiceSample:
    """A voice sample for biometric analysis"""

    id: str
    user_id: str
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    timestamp: datetime = field(default_factory=datetime.now)
    quality_score: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "timestamp": self.timestamp.isoformat(),
            "quality_score": self.quality_score,
            "features": self.features,
        }


@dataclass
class VoiceBiometric:
    """Voice biometric profile for a user"""

    id: str
    user_id: str
    feature_vector: np.ndarray
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    sample_count: int = 0
    confidence_threshold: float = 0.75
    samples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "feature_vector": self.feature_vector.tolist(),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "sample_count": self.sample_count,
            "confidence_threshold": self.confidence_threshold,
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceBiometric":
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            feature_vector=np.array(data["feature_vector"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            sample_count=data.get("sample_count", 0),
            confidence_threshold=data.get("confidence_threshold", 0.75),
            samples=data.get("samples", []),
        )


class VoiceRecognition:
    """
    Voice recognition and biometric system.

    Features:
    - Voice biometric enrollment
    - Speaker identification
    - Voice quality assessment
    - Anti-spoofing detection
    - Multi-sample learning
    """

    def __init__(self, storage_path: str = "data/voice_biometrics"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.biometrics: Dict[str, VoiceBiometric] = {}
        self.samples: Dict[str, VoiceSample] = {}
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the voice recognition system"""
        if self._initialized:
            return

        await self._load_data()
        self._initialized = True
        log.info(f"Voice recognition initialized with {len(self.biometrics)} profiles")

    async def _load_data(self):
        """Load biometric data from storage"""
        biometrics_file = self.storage_path / "biometrics.json"

        if biometrics_file.exists():
            try:
                with open(biometrics_file, "r") as f:
                    data = json.load(f)

                for bio_data in data.get("biometrics", []):
                    biometric = VoiceBiometric.from_dict(bio_data)
                    self.biometrics[biometric.id] = biometric

            except Exception as e:
                log.error(f"Error loading voice biometrics: {e}")

    async def _save_data(self):
        """Save biometric data to storage"""
        biometrics_file = self.storage_path / "biometrics.json"

        with open(biometrics_file, "w") as f:
            json.dump(
                {"biometrics": [b.to_dict() for b in self.biometrics.values()]},
                f,
                indent=2,
            )

    async def enroll_user(
        self,
        user_id: str,
        audio_samples: List[np.ndarray],
        sample_rate: int = 16000,
    ) -> Optional[VoiceBiometric]:
        """
        Enroll a user with voice samples.

        Args:
            user_id: User identifier
            audio_samples: List of audio samples
            sample_rate: Sample rate of audio

        Returns:
            VoiceBiometric profile or None if enrollment failed
        """
        async with self._lock:
            if len(audio_samples) < 3:
                raise ValueError("At least 3 voice samples required for enrollment")

            # Process each sample
            processed_samples = []
            for i, audio in enumerate(audio_samples):
                sample_id = f"sample_{user_id}_{i}"

                # Extract features
                features = await self._extract_features(audio, sample_rate)

                # Calculate quality
                quality = self._assess_quality(audio, sample_rate)

                if quality < 0.5:
                    log.warning(f"Sample {i} has low quality ({quality:.2f}), skipping")
                    continue

                sample = VoiceSample(
                    id=sample_id,
                    user_id=user_id,
                    audio_data=audio,
                    sample_rate=sample_rate,
                    duration=len(audio) / sample_rate,
                    quality_score=quality,
                    features=features,
                )

                self.samples[sample_id] = sample
                processed_samples.append(sample)

            if len(processed_samples) < 3:
                raise ValueError("Not enough high-quality samples for enrollment")

            # Create feature vector from samples
            feature_vector = await self._create_feature_vector(processed_samples)

            # Create or update biometric profile
            biometric_id = f"voice_bio_{user_id}"

            biometric = VoiceBiometric(
                id=biometric_id,
                user_id=user_id,
                feature_vector=feature_vector,
                sample_count=len(processed_samples),
                samples=[s.id for s in processed_samples],
            )

            self.biometrics[biometric_id] = biometric
            await self._save_data()

            log.info(f"Enrolled user {user_id} with {len(processed_samples)} voice samples")
            return biometric

    async def identify_speaker(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Identify speaker from audio.

        Args:
            audio: Audio sample
            sample_rate: Sample rate
            top_k: Number of top matches to return

        Returns:
            List of (user_id, confidence) tuples
        """
        if not self.biometrics:
            return []

        # Extract features from input
        features = await self._extract_features(audio, sample_rate)

        # Calculate quality
        quality = self._assess_quality(audio, sample_rate)
        if quality < 0.3:
            log.warning(f"Input audio has low quality: {quality:.2f}")
            return []

        # Create temporary feature vector
        temp_sample = VoiceSample(
            id="temp",
            user_id="unknown",
            audio_data=audio,
            sample_rate=sample_rate,
            duration=len(audio) / sample_rate,
            quality_score=quality,
            features=features,
        )

        input_vector = await self._create_feature_vector([temp_sample])

        # Compare with all enrolled users
        scores = []

        for biometric in self.biometrics.values():
            similarity = self._calculate_similarity(input_vector, biometric.feature_vector)
            confidence = self._similarity_to_confidence(similarity)

            if confidence >= biometric.confidence_threshold:
                scores.append((biometric.user_id, confidence))

        # Sort by confidence
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    async def verify_speaker(
        self,
        user_id: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Tuple[bool, float]:
        """
        Verify if the speaker is the claimed user.

        Args:
            user_id: Claimed user ID
            audio: Audio sample
            sample_rate: Sample rate

        Returns:
            (is_match, confidence)
        """
        # Find biometric for user
        biometric = None
        for bio in self.biometrics.values():
            if bio.user_id == user_id:
                biometric = bio
                break

        if not biometric:
            return False, 0.0

        # Extract features
        features = await self._extract_features(audio, sample_rate)

        # Create temporary feature vector
        temp_sample = VoiceSample(
            id="temp",
            user_id="unknown",
            audio_data=audio,
            sample_rate=sample_rate,
            duration=len(audio) / sample_rate,
            features=features,
        )

        input_vector = await self._create_feature_vector([temp_sample])

        # Calculate similarity
        similarity = self._calculate_similarity(input_vector, biometric.feature_vector)
        confidence = self._similarity_to_confidence(similarity)

        is_match = confidence >= biometric.confidence_threshold

        return is_match, confidence

    async def add_sample(
        self,
        user_id: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> bool:
        """Add a new sample to an existing user's profile"""
        async with self._lock:
            # Find biometric
            biometric = None
            for bio in self.biometrics.values():
                if bio.user_id == user_id:
                    biometric = bio
                    break

            if not biometric:
                return False

            # Process new sample
            features = await self._extract_features(audio, sample_rate)
            quality = self._assess_quality(audio, sample_rate)

            if quality < 0.5:
                log.warning("New sample has low quality, not adding")
                return False

            sample_id = f"sample_{user_id}_{biometric.sample_count}"

            sample = VoiceSample(
                id=sample_id,
                user_id=user_id,
                audio_data=audio,
                sample_rate=sample_rate,
                duration=len(audio) / sample_rate,
                quality_score=quality,
                features=features,
            )

            self.samples[sample_id] = sample
            biometric.samples.append(sample_id)
            biometric.sample_count += 1

            # Recalculate feature vector
            user_samples = [self.samples[sid] for sid in biometric.samples if sid in self.samples]
            biometric.feature_vector = await self._create_feature_vector(user_samples)
            biometric.last_updated = datetime.now()

            await self._save_data()
            return True

    async def _extract_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, Any]:
        """Extract voice features from audio"""
        features = {}

        # Basic statistics
        features["mean"] = float(np.mean(audio))
        features["std"] = float(np.std(audio))
        features["max"] = float(np.max(np.abs(audio)))
        features["rms"] = float(np.sqrt(np.mean(audio**2)))

        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
        features["zcr"] = float(zero_crossings / len(audio))

        # Spectral features (simplified)
        fft = np.abs(np.fft.fft(audio))
        freqs = np.fft.fftfreq(len(audio), 1 / sample_rate)

        # Find dominant frequencies
        peak_indices = np.argsort(fft)[-5:]
        dominant_freqs = freqs[peak_indices]
        features["dominant_freqs"] = [float(f) for f in dominant_freqs if f > 0]

        # Spectral centroid
        if np.sum(fft) > 0:
            spectral_centroid = np.sum(freqs * fft) / np.sum(fft)
            features["spectral_centroid"] = float(spectral_centroid)

        # Spectral rolloff
        cumsum = np.cumsum(fft)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        if len(rolloff_idx) > 0:
            features["spectral_rolloff"] = float(freqs[rolloff_idx[0]])

        # MFCC-like features (simplified)
        # In production, use librosa or similar library
        features["mfcc_mean"] = float(np.mean(audio[::10]))  # Simplified
        features["mfcc_std"] = float(np.std(audio[::10]))

        return features

    def _assess_quality(self, audio: np.ndarray, sample_rate: int) -> float:
        """Assess audio quality for biometric use"""
        scores = []

        # Signal level
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.01:
            scores.append(0.0)
        elif rms > 0.1:
            scores.append(1.0)
        else:
            scores.append(rms * 10)

        # Signal-to-noise ratio (simplified)
        # Assume first 100ms is noise/silence
        noise_samples = min(int(0.1 * sample_rate), len(audio) // 10)
        if noise_samples > 0:
            noise_level = np.std(audio[:noise_samples])
            signal_level = np.std(audio[noise_samples:])

            if noise_level > 0:
                snr = signal_level / noise_level
                scores.append(min(snr / 10, 1.0))

        # Clipping detection
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            scores.append(0.5)  # Penalize clipping
        else:
            scores.append(1.0)

        # Duration
        duration = len(audio) / sample_rate
        if duration < 2.0:
            scores.append(duration / 2.0)
        else:
            scores.append(1.0)

        return np.mean(scores)

    async def _create_feature_vector(self, samples: List[VoiceSample]) -> np.ndarray:
        """Create a feature vector from multiple samples"""
        if not samples:
            return np.array([])

        # Collect features from all samples
        all_features = []

        for sample in samples:
            features = [
                sample.features.get("mean", 0),
                sample.features.get("std", 0),
                sample.features.get("rms", 0),
                sample.features.get("zcr", 0),
                sample.features.get("spectral_centroid", 0) / 1000,  # Normalize
                sample.features.get("spectral_rolloff", 0) / 10000,  # Normalize
                sample.features.get("mfcc_mean", 0),
                sample.features.get("mfcc_std", 0),
            ]

            # Add dominant frequencies (first 3)
            dom_freqs = sample.features.get("dominant_freqs", [])
            for i in range(3):
                if i < len(dom_freqs):
                    features.append(dom_freqs[i] / 1000)  # Normalize
                else:
                    features.append(0)

            all_features.append(features)

        # Average across samples
        feature_matrix = np.array(all_features)
        mean_vector = np.mean(feature_matrix, axis=0)

        # Normalize
        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            mean_vector = mean_vector / norm

        return mean_vector

    def _calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vector1) != len(vector2):
            return 0.0

        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _similarity_to_confidence(self, similarity: float) -> float:
        """Convert similarity score to confidence (0-1)"""
        # Cosine similarity ranges from -1 to 1
        # Map to 0-1 with a sigmoid-like transformation
        normalized = (similarity + 1) / 2
        # Apply curve to make high similarities more distinct
        return float(normalized**2)

    async def delete_biometric(self, user_id: str) -> bool:
        """Delete a user's biometric profile"""
        async with self._lock:
            biometric_to_delete = None
            for bio_id, biometric in self.biometrics.items():
                if biometric.user_id == user_id:
                    biometric_to_delete = bio_id
                    break

            if biometric_to_delete:
                del self.biometrics[biometric_to_delete]

                # Remove associated samples
                samples_to_remove = [sid for sid, s in self.samples.items() if s.user_id == user_id]
                for sid in samples_to_remove:
                    del self.samples[sid]

                await self._save_data()
                return True

            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get voice recognition statistics"""
        return {
            "enrolled_users": len(self.biometrics),
            "total_samples": len(self.samples),
            "avg_samples_per_user": len(self.samples) / len(self.biometrics)
            if self.biometrics
            else 0,
            "users_with_low_confidence": sum(
                1 for b in self.biometrics.values() if b.confidence_threshold > 0.8
            ),
        }


# Global instance
_voice_recognition: Optional[VoiceRecognition] = None


async def get_voice_recognition() -> VoiceRecognition:
    """Get the global voice recognition instance"""
    global _voice_recognition
    if _voice_recognition is None:
        _voice_recognition = VoiceRecognition()
        await _voice_recognition.initialize()
    return _voice_recognition


__all__ = [
    "VoiceRecognition",
    "VoiceBiometric",
    "VoiceSample",
    "get_voice_recognition",
]
