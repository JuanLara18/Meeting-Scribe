# processing/diarize.py
# ───────────────────────────────────────────────────────────────────
# Custom speaker diarization using signal processing and clustering
# Produces "who spoke when" time intervals

import os
import logging
import time
import numpy as np
import librosa
import soundfile as sf
from typing import List, Dict, Any, Optional
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

# Suppress specific warnings that might occur during processing
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

class SpeakerDiarizer:
    """
    Segments audio into speaker turns using audio features and clustering.
    """

    def __init__(
        self,
        model_name: str = None,  # Kept for API compatibility
        use_auth_token: str = None,  # Kept for API compatibility
        device: str = "cpu"
    ):
        """
        Args:
          model_name    – Ignored (kept for API compatibility)
          use_auth_token– Ignored (kept for API compatibility)
          device        – "cpu" or "cuda" (affects feature extraction speed)
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters for diarization
        self.frame_length = 0.025  # 25ms frame
        self.frame_shift = 0.010   # 10ms hop
        self.n_mfcc = 20           # Number of MFCC coefficients
        self.min_segment_dur = 0.3  # Minimum segment duration in seconds (reduced for better granularity)
        self.vad_threshold = 0.25   # Energy threshold for VAD (tuned for better detection)
        self.vad_pad_dur = 0.15     # Padding duration for VAD segments in seconds (increased for context)
        self.merge_threshold = 0.3  # Threshold for merging adjacent segments from same speaker
    
    def diarize(
        self,
        wav_path: str,
        min_speakers: int = None,
        max_speakers: int = None
    ) -> List[Dict[str, Any]]:
        """
        Run speaker diarization using MFCC features and clustering.

        Args:
          wav_path     – path to 16 kHz mono WAV
          min_speakers – optional lower bound on # of speakers
          max_speakers – optional upper bound on # of speakers

        Returns:
          A list of speaker segments:
            [
              {
                "speaker": "SPEAKER_00",
                "start":   0.00,
                "end":     1.75
              },
              …
            ]

        Raises:
          FileNotFoundError if wav_path missing
          RuntimeError on processing failure
        """
        # Check if input file exists
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"Input WAV file not found: {wav_path}")
        
        # Log start of diarization
        self.logger.info(f"Starting custom diarization of {wav_path}")
        
        # Prepare speaker constraints
        if min_speakers is None:
            min_speakers = 1
        if max_speakers is None:
            max_speakers = 10  # Reasonable default upper limit
            
        self.logger.debug(f"Speaker constraints: min={min_speakers}, max={max_speakers}")
        
        start_time = time.time()
        
        try:
            # 1. Load and preprocess audio
            y, sr = librosa.load(wav_path, sr=None)
            self.logger.info(f"Loaded audio: {len(y)/sr:.2f}s at {sr}Hz")
            
            # 2. Perform Voice Activity Detection
            speech_segments = self._detect_speech(y, sr)
            self.logger.info(f"Detected {len(speech_segments)} speech segments")
            
            if not speech_segments:
                self.logger.warning("No speech detected in audio")
                return []
            
            # 3. Extract features from speech segments
            embeddings, segment_info = self._extract_features(y, sr, speech_segments)
            
            if len(segment_info) == 0:
                self.logger.warning("No valid features extracted")
                return []
                
            # 4. Determine number of speakers through clustering
            num_speakers = self._estimate_num_speakers(embeddings, min_speakers, max_speakers)
            self.logger.info(f"Estimated number of speakers: {num_speakers}")
            
            # 5. Perform speaker clustering
            speaker_labels = self._cluster_speakers(embeddings, num_speakers)
            
            # 6. Create final speaker segments
            segments = self._create_speaker_segments(segment_info, speaker_labels)
            
            # 7. Post-process segments (merge short segments, handle overlaps)
            segments = self._post_process_segments(segments)
            
            # Log completion
            diarize_time = time.time() - start_time
            self.logger.info(f"Diarization completed in {diarize_time:.2f} seconds")
            self.logger.info(f"Found {num_speakers} speakers across {len(segments)} segments")
            
            return segments
            
        except Exception as e:
            self.logger.error(f"Diarization failed: {str(e)}")
            raise RuntimeError(f"Custom diarization failed: {str(e)}")
    
    def _detect_speech(self, y: np.ndarray, sr: int) -> List[Dict[str, float]]:
        """
        Detects speech segments using energy-based Voice Activity Detection.
        """
        self.logger.debug("Performing Voice Activity Detection")
        
        # Calculate frame parameters
        frame_length_samples = int(self.frame_length * sr)
        frame_shift_samples = int(self.frame_shift * sr)
        
        # Calculate energy in each frame
        energy = librosa.feature.rms(
            y=y, 
            frame_length=frame_length_samples, 
            hop_length=frame_shift_samples
        )[0]
        
        # Normalize energy to 0-1 range
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
        
        # Apply threshold for speech detection
        is_speech = energy_norm > self.vad_threshold
        
        # Convert frame-level decisions to time segments
        segments = []
        in_speech = False
        speech_start = 0
        
        for i, speech in enumerate(is_speech):
            time_sec = i * frame_shift_samples / sr
            
            if speech and not in_speech:
                # Start of speech segment
                in_speech = True
                speech_start = max(0, time_sec - self.vad_pad_dur)  # Add padding
            
            elif not speech and in_speech:
                # End of speech segment
                in_speech = False
                speech_end = min(len(y)/sr, time_sec + self.vad_pad_dur)  # Add padding
                
                # Only add if segment is long enough
                if speech_end - speech_start >= self.min_segment_dur:
                    segments.append({
                        'start': speech_start,
                        'end': speech_end
                    })
        
        # Handle the case where audio ends during speech
        if in_speech:
            speech_end = len(y) / sr
            if speech_end - speech_start >= self.min_segment_dur:
                segments.append({
                    'start': speech_start,
                    'end': speech_end
                })
        
        return segments
    
    def _extract_features(self, y: np.ndarray, sr: int, segments: List[Dict[str, float]]) -> tuple:
        """
        Extracts comprehensive audio features from segments including:
        - MFCCs (Mel-frequency cepstral coefficients)
        - Spectral features (centroid, rolloff, contrast)
        - Pitch and energy features
        """
        self.logger.debug("Extracting enhanced audio features")

        # Initialize lists to store features and segment info
        all_embeddings = []
        segment_info = []

        # For each speech segment
        for i, segment in enumerate(segments):
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)

            # Extract segment audio
            segment_audio = y[start_sample:end_sample]

            # Skip if segment is too short
            if len(segment_audio) < sr * self.min_segment_dur:
                continue

            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=segment_audio,
                sr=sr,
                n_mfcc=self.n_mfcc,
                hop_length=int(self.frame_shift * sr),
                n_fft=int(self.frame_length * sr)
            )

            # Add delta and delta-delta features for better speaker discrimination
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)

            # Extract spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=segment_audio,
                sr=sr,
                hop_length=int(self.frame_shift * sr),
                n_fft=int(self.frame_length * sr)
            )

            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=segment_audio,
                sr=sr,
                hop_length=int(self.frame_shift * sr),
                n_fft=int(self.frame_length * sr)
            )

            spectral_contrast = librosa.feature.spectral_contrast(
                y=segment_audio,
                sr=sr,
                hop_length=int(self.frame_shift * sr),
                n_fft=int(self.frame_length * sr)
            )

            # Extract zero crossing rate (useful for voice distinction)
            zcr = librosa.feature.zero_crossing_rate(
                segment_audio,
                frame_length=int(self.frame_length * sr),
                hop_length=int(self.frame_shift * sr)
            )

            # Extract chroma features (pitch content)
            chroma = librosa.feature.chroma_stft(
                y=segment_audio,
                sr=sr,
                hop_length=int(self.frame_shift * sr),
                n_fft=int(self.frame_length * sr)
            )

            # Combine all features
            features = np.vstack([
                mfccs,
                delta_mfccs,
                delta2_mfccs,
                spectral_centroid,
                spectral_rolloff,
                spectral_contrast,
                zcr,
                chroma
            ]).T

            # Handle edge case of very short segments
            if features.shape[0] == 0:
                continue

            # Compute segment embedding by averaging frame-level features
            embedding = np.mean(features, axis=0)

            all_embeddings.append(embedding)
            segment_info.append({
                'start': segment['start'],
                'end': segment['end'],
                'index': i
            })

        # Convert to numpy array
        if all_embeddings:
            embeddings_array = np.vstack(all_embeddings)

            # Normalize features
            scaler = StandardScaler()
            embeddings_array = scaler.fit_transform(embeddings_array)

            return embeddings_array, segment_info
        else:
            return np.array([]), []
    
    def _estimate_num_speakers(self, embeddings: np.ndarray, min_speakers: int, max_speakers: int) -> int:
        """
        Estimates the optimal number of speakers using silhouette score and elbow method.
        """
        self.logger.debug("Estimating number of speakers")

        # Handle edge cases
        if len(embeddings) <= min_speakers:
            return min_speakers

        # If only one possible value, return it
        if min_speakers == max_speakers:
            return min_speakers

        # Limit max_speakers to number of segments
        max_speakers = min(max_speakers, len(embeddings))

        # For very few embeddings, use simpler approach
        if len(embeddings) < 4:
            return min(min_speakers, len(embeddings))

        # Try different numbers of clusters and compute metrics
        silhouette_scores = []
        distortions = []
        possible_clusters = range(max(2, min_speakers), min(max_speakers + 1, len(embeddings)))

        for n_clusters in possible_clusters:
            # Use try-except block to handle different scikit-learn versions
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    affinity='euclidean',
                    linkage='ward'
                )
            except TypeError:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward'
                )

            labels = clustering.fit_predict(embeddings)

            # Calculate silhouette score (higher is better)
            try:
                score = silhouette_score(embeddings, labels)
                silhouette_scores.append(score)
            except:
                silhouette_scores.append(-1)

            # Calculate distortion (within-cluster sum of squares)
            distortion = 0
            for i in range(n_clusters):
                cluster_points = embeddings[labels == i]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    distortion += np.sum(np.linalg.norm(cluster_points - centroid, axis=1)**2)

            distortions.append(distortion)

        # Find optimal number using silhouette score
        if silhouette_scores and max(silhouette_scores) > 0:
            best_idx = np.argmax(silhouette_scores)
            best_n_clusters = possible_clusters[best_idx]
            self.logger.debug(f"Best silhouette score: {silhouette_scores[best_idx]:.3f} for {best_n_clusters} speakers")
            return best_n_clusters

        # Fallback to elbow method if silhouette fails
        if len(distortions) > 1:
            # Calculate rate of change
            diffs = np.diff(distortions)

            # Normalize differences
            if np.max(np.abs(diffs)) > 0:
                diffs = diffs / np.max(np.abs(diffs))

            # Look for elbow point (where difference becomes small)
            elbow_threshold = 0.15
            for i, diff in enumerate(diffs):
                if abs(diff) < elbow_threshold:
                    return possible_clusters[i+1]

        # Default fallback
        return min(max(2, min_speakers), max_speakers)
    
    def _cluster_speakers(self, embeddings: np.ndarray, num_speakers: int) -> np.ndarray:
        """
        Clusters segment embeddings into speaker groups.
        """
        self.logger.debug(f"Clustering segments into {num_speakers} speakers")
        
        # Handle edge case
        if len(embeddings) <= 1:
            return np.zeros(len(embeddings), dtype=int)
            
        # Use try-except block to handle different scikit-learn versions
        try:
            # Try with both parameters (newer scikit-learn versions)
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers, 
                affinity='euclidean', 
                linkage='ward'
            )
        except TypeError:
            # Fallback for older versions or different parameter structure
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                linkage='ward'
            )
        
        labels = clustering.fit_predict(embeddings)
        
        return labels
    
    def _create_speaker_segments(self, segment_info: List[Dict[str, Any]], speaker_labels: np.ndarray) -> List[Dict[str, Any]]:
        """
        Creates speaker segments from clustering results.
        """
        self.logger.debug("Creating speaker segments from clustering results")
        
        # Create segments with speaker labels
        segments = []
        for i, segment in enumerate(segment_info):
            if i < len(speaker_labels):
                speaker_id = int(speaker_labels[i])
                segments.append({
                    'speaker': f"SPEAKER_{speaker_id:02d}",
                    'start': segment['start'],
                    'end': segment['end']
                })
        
        # Sort by start time
        segments.sort(key=lambda s: s['start'])
        
        return segments
    
    def _post_process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Post-processes segments to merge those from the same speaker and filter very short segments.
        """
        self.logger.debug("Post-processing speaker segments")

        if not segments:
            return []

        # Sort by start time
        sorted_segments = sorted(segments, key=lambda s: s['start'])

        # Merge adjacent segments from the same speaker
        merged = []
        current = sorted_segments[0].copy()

        for segment in sorted_segments[1:]:
            # If same speaker and gap is within merge threshold, merge
            gap = segment['start'] - current['end']
            if segment['speaker'] == current['speaker'] and gap < self.merge_threshold:
                current['end'] = segment['end']
            else:
                # Only add segment if it meets minimum duration
                if current['end'] - current['start'] >= self.min_segment_dur:
                    merged.append(current)
                current = segment.copy()

        # Add the last segment if it meets minimum duration
        if current['end'] - current['start'] >= self.min_segment_dur:
            merged.append(current)

        self.logger.debug(f"Post-processing: {len(sorted_segments)} -> {len(merged)} segments")

        return merged