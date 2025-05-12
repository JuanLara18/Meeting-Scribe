# processing/merge.py
# ───────────────────────────────────────────────────────────────────
# Aligns Whisper's transcript segments with diarization turns,
# producing unified speaker-labeled text entries.

import logging
from typing import List, Dict, Any, Optional, Tuple

class SegmentMerger:
    """
    Merges ASR segments and diarization segments into unified speaker-labelled entries.
    """

    def __init__(
        self,
        max_gap: float = 0.5,
        min_overlap: float = 0.1
    ):
        """
        Args:
          max_gap     – maximum seconds allowed between ASR and speaker turn to still align
          min_overlap – minimum overlap in seconds required to consider segments matching
        """
        self.max_gap = max_gap
        self.min_overlap = min_overlap
        self.logger = logging.getLogger(__name__)
    
    def _compute_overlap(self, segment1: Dict[str, Any], segment2: Dict[str, Any]) -> float:
        """
        Calculate the overlap duration between two time segments.
        
        Args:
            segment1: First segment with 'start' and 'end' keys
            segment2: Second segment with 'start' and 'end' keys
            
        Returns:
            Overlap duration in seconds (0 if no overlap)
        """
        # Get start and end times
        start1, end1 = segment1["start"], segment1["end"]
        start2, end2 = segment2["start"], segment2["end"]
        
        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap = max(0, overlap_end - overlap_start)
        
        return overlap
    
    def _find_best_speaker(
        self, 
        transcript_segment: Dict[str, Any], 
        diarization_segments: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Find the best matching speaker for a transcript segment.
        
        Args:
            transcript_segment: Whisper segment with 'start' and 'end' keys
            diarization_segments: List of speaker segments
            
        Returns:
            Best matching speaker ID or None if no match found
        """
        best_speaker = None
        best_overlap = 0
        
        for speaker_segment in diarization_segments:
            # Calculate overlap
            overlap = self._compute_overlap(transcript_segment, speaker_segment)
            
            # Update best match if this is better
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_segment["speaker"]
        
        # Only consider it a match if overlap exceeds minimum threshold
        if best_overlap < self.min_overlap:
            # Try a proximity-based match if no overlap found
            return self._find_nearest_speaker(transcript_segment, diarization_segments)
            
        return best_speaker
    
    def _find_nearest_speaker(
        self, 
        segment: Dict[str, Any], 
        diarization_segments: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Find the nearest speaker by time proximity when no overlap exists.
        
        Args:
            segment: Transcript segment
            diarization_segments: List of speaker segments
            
        Returns:
            Nearest speaker ID or None if none found within max_gap
        """
        segment_mid = (segment["start"] + segment["end"]) / 2
        nearest_speaker = None
        min_distance = float('inf')
        
        for speaker_segment in diarization_segments:
            speaker_mid = (speaker_segment["start"] + speaker_segment["end"]) / 2
            distance = abs(segment_mid - speaker_mid)
            
            if distance < min_distance:
                min_distance = distance
                nearest_speaker = speaker_segment["speaker"]
        
        # Only return a speaker if within max_gap
        if min_distance <= self.max_gap:
            return nearest_speaker
        
        return None

    def merge(
        self,
        transcript_segments: List[Dict[str, Any]],
        diarization_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Aligns and merges two lists of segments.

        Args:
          transcript_segments – Whisper output `.segments`, each:
            {"id": int, "start": float, "end": float, "text": str}
          diarization_segments – pyannote output, each:
            {"speaker": str, "start": float, "end": float}

        Returns:
          A list of merged entries:
          [
            {
              "speaker":  "SPEAKER_00",
              "start":    12.34,
              "end":      15.67,
              "text":     "Transcribed text..."
            },
            …
          ]

        Raises:
          ValueError if inputs are empty or cannot be aligned
        """
        # Validate inputs
        if not transcript_segments:
            raise ValueError("Empty transcript segments list provided")
        
        if not diarization_segments:
            raise ValueError("Empty diarization segments list provided")
        
        self.logger.info(f"Merging {len(transcript_segments)} transcript segments with {len(diarization_segments)} speaker segments")
        
        # Initialize merged segments list
        merged_segments = []
        
        # Keep track of segments without speaker match
        unassigned_segments = []
        
        # Process each transcript segment
        for transcript_segment in transcript_segments:
            # Find best matching speaker
            speaker = self._find_best_speaker(transcript_segment, diarization_segments)
            
            # Create merged segment
            merged_segment = {
                "start": transcript_segment["start"],
                "end": transcript_segment["end"],
                "text": transcript_segment["text"].strip()
            }
            
            # Add speaker if found
            if speaker:
                merged_segment["speaker"] = speaker
                merged_segments.append(merged_segment)
            else:
                # No speaker match found - add to unassigned list
                unassigned_segments.append(merged_segment)
                self.logger.debug(f"No speaker match for segment [{merged_segment['start']:.2f}-{merged_segment['end']:.2f}]: {merged_segment['text'][:30]}...")
        
        # Assign unassigned segments using surrounding context
        if unassigned_segments:
            self._assign_remaining_segments(merged_segments, unassigned_segments)
        
        # Sort all segments by start time
        all_segments = merged_segments + unassigned_segments
        all_segments.sort(key=lambda x: x["start"])
        
        # Log merge results
        total_speakers = len(set(s.get("speaker", "UNKNOWN") for s in all_segments))
        self.logger.info(f"Merged into {len(all_segments)} segments with {total_speakers} speakers")
        
        unknown_count = sum(1 for s in all_segments if s.get("speaker") == "UNKNOWN")
        if unknown_count:
            self.logger.warning(f"{unknown_count} segments ({unknown_count/len(all_segments):.1%}) have UNKNOWN speaker assignment")
        
        return all_segments
    
    def _assign_remaining_segments(
        self, 
        merged_segments: List[Dict[str, Any]], 
        unassigned_segments: List[Dict[str, Any]]
    ) -> None:
        """
        Try to assign speakers to unassigned segments based on temporal proximity.
        
        Args:
            merged_segments: List of segments with assigned speakers
            unassigned_segments: List of segments without speakers
        
        Note:
            This function modifies both lists in place.
        """
        # Skip if we have no reference segments with speakers
        if not merged_segments:
            # When no segments have speakers, mark all as UNKNOWN
            for segment in unassigned_segments:
                segment["speaker"] = "UNKNOWN"
            return
            
        # Sort segments by time for efficient lookup
        merged_segments.sort(key=lambda x: x["start"])
        unassigned_segments.sort(key=lambda x: x["start"])
        
        # Process each unassigned segment
        assigned_indices = []
        
        for i, segment in enumerate(unassigned_segments):
            # Find closest segments before and after
            prev_segment = None
            next_segment = None
            
            # Find previous segment
            for merged in reversed(merged_segments):
                if merged["end"] <= segment["start"]:
                    prev_segment = merged
                    break
            
            # Find next segment
            for merged in merged_segments:
                if merged["start"] >= segment["end"]:
                    next_segment = merged
                    break
            
            # Determine which one is closer
            if prev_segment and next_segment:
                gap_to_prev = segment["start"] - prev_segment["end"]
                gap_to_next = next_segment["start"] - segment["end"]
                
                if gap_to_prev <= gap_to_next and gap_to_prev <= self.max_gap:
                    segment["speaker"] = prev_segment["speaker"]
                    assigned_indices.append(i)
                    merged_segments.append(segment)  # Add to merged list
                elif gap_to_next <= self.max_gap:
                    segment["speaker"] = next_segment["speaker"]
                    assigned_indices.append(i)
                    merged_segments.append(segment)  # Add to merged list
            
            # Only previous segment exists
            elif prev_segment and (segment["start"] - prev_segment["end"] <= self.max_gap):
                segment["speaker"] = prev_segment["speaker"]
                assigned_indices.append(i)
                merged_segments.append(segment)  # Add to merged list
            
            # Only next segment exists
            elif next_segment and (next_segment["start"] - segment["end"] <= self.max_gap):
                segment["speaker"] = next_segment["speaker"]
                assigned_indices.append(i)
                merged_segments.append(segment)  # Add to merged list
        
        # Remove assigned segments from unassigned list
        for index in sorted(assigned_indices, reverse=True):
            del unassigned_segments[index]
        
        # For any remaining segments, assign an "UNKNOWN" speaker
        for segment in unassigned_segments:
            segment["speaker"] = "UNKNOWN"