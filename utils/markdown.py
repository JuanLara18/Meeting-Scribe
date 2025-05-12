# utils/markdown.py
# ───────────────────────────────────────────────────────────────────
# Exports merged speaker/text segments into:
# 1) a raw JSON dump, and
# 2) a human-readable Markdown transcript grouped by time blocks.

import os
import json
import logging
from typing import List, Dict, Any, Optional

class MarkdownExporter:
    """
    Generates a Markdown file with timestamped, speaker-labelled entries and minute-based summaries.
    """

    def __init__(
        self,
        output_md: str = "results/transcript.md",
        output_json: str = "results/transcript.json"
    ):
        """
        Args:
          output_md   – filepath for the generated Markdown
          output_json– filepath for raw JSON dump of merged segments
        """
        self.output_md = output_md
        self.output_json = output_json
        self.logger = logging.getLogger(__name__)
    
    def _ensure_directory_exists(self, filepath: str) -> None:
        """
        Create directory for the file if it doesn't exist.
        
        Args:
          filepath – path to file
        """
        directory = os.path.dirname(os.path.abspath(filepath))
        if directory and not os.path.exists(directory):
            self.logger.debug(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds as HH:MM:SS.
        
        Args:
          seconds – time in seconds
          
        Returns:
          Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def export_json(
        self,
        merged_segments: List[Dict[str, Any]]
    ) -> None:
        """
        Save the merged segments list to JSON.

        Args:
          merged_segments – list from SegmentMerger.merge()
          
        Raises:
          ValueError if merged_segments is invalid
          RuntimeError if writing fails
        """
        if not merged_segments:
            raise ValueError("Cannot export empty segments list")
        
        self.logger.info(f"Exporting {len(merged_segments)} segments to JSON: {self.output_json}")
        
        try:
            # Create directory if needed
            self._ensure_directory_exists(self.output_json)
            
            # Write JSON file
            with open(self.output_json, 'w', encoding='utf-8') as f:
                json.dump(merged_segments, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"JSON export complete: {self.output_json}")
        except Exception as e:
            self.logger.error(f"Failed to export JSON: {str(e)}")
            raise RuntimeError(f"Failed to export JSON: {str(e)}")
    
    def export_markdown(
        self,
        merged_segments: List[Dict[str, Any]],
        block_minutes: int = 1
    ) -> None:
        """
        Write a Markdown document grouping entries into minute blocks.

        Args:
          merged_segments – list from SegmentMerger.merge()
          block_minutes   – size of each time block (in minutes)
          
        Raises:
          ValueError if merged_segments is invalid or block_minutes <= 0
          RuntimeError if writing fails
        """
        if not merged_segments:
            raise ValueError("Cannot export empty segments list")
        
        if block_minutes <= 0:
            raise ValueError(f"Invalid block_minutes: {block_minutes}. Must be positive.")
        
        self.logger.info(f"Exporting {len(merged_segments)} segments to Markdown: {self.output_md}")
        
        try:
            # Create directory if needed
            self._ensure_directory_exists(self.output_md)
            
            # Group segments by time blocks
            blocks = self._group_by_time_blocks(merged_segments, block_minutes)
            
            # Write Markdown file
            with open(self.output_md, 'w', encoding='utf-8') as f:
                # Write header with meeting metadata
                f.write("# Meeting Transcript\n\n")
                
                # Get meeting duration
                if merged_segments:
                    duration = merged_segments[-1]["end"]
                    duration_formatted = self._format_timestamp(duration)
                    f.write(f"**Duration:** {duration_formatted}\n\n")
                    
                    # Count unique speakers
                    speakers = set(s.get("speaker", "UNKNOWN") for s in merged_segments)
                    f.write(f"**Participants:** {len(speakers)}\n\n")
                
                f.write("---\n\n")
                
                # Write each time block
                for block_start in sorted(blocks.keys()):
                    block_segments = blocks[block_start]
                    block_start_mins = block_start * block_minutes
                    block_end_mins = block_start_mins + block_minutes
                    
                    # Format block header with time range
                    start_formatted = self._format_timestamp(block_start_mins * 60)
                    end_formatted = self._format_timestamp(block_end_mins * 60)
                    f.write(f"## {start_formatted} – {end_formatted}\n\n")
                    
                    # Write each segment in the block
                    for segment in block_segments:
                        speaker = segment.get("speaker", "UNKNOWN")
                        start = self._format_timestamp(segment["start"])
                        text = segment["text"].strip()
                        
                        # Format with speaker name bold, timestamp in code block
                        f.write(f"**{speaker}** `[{start}]`: {text}\n\n")
                
                # Write footer
                f.write("---\n\n")
                f.write("*Generated by MeetingScribe*\n")
            
            self.logger.info(f"Markdown export complete: {self.output_md}")
        except Exception as e:
            self.logger.error(f"Failed to export Markdown: {str(e)}")
            raise RuntimeError(f"Failed to export Markdown: {str(e)}")
    
    def _group_by_time_blocks(
        self,
        merged_segments: List[Dict[str, Any]],
        block_minutes: int
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Group segments into time blocks based on their start times.
        
        Args:
          merged_segments – list of speaker segments with timestamps
          block_minutes   – size of each block in minutes
          
        Returns:
          Dictionary mapping block index to segment list
        """
        blocks: Dict[int, List[Dict[str, Any]]] = {}
        
        for segment in merged_segments:
            # Calculate which block this segment belongs to
            # Convert seconds to minutes, then find which N-minute block it falls into
            mins = segment["start"] / 60
            block_index = int(mins // block_minutes)
            
            # Add segment to appropriate block
            if block_index not in blocks:
                blocks[block_index] = []
            blocks[block_index].append(segment)
        
        # Ensure segments within each block are sorted by start time
        for block_index in blocks:
            blocks[block_index].sort(key=lambda s: s["start"])
        
        return blocks