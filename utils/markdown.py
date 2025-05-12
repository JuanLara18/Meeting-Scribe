# utils/markdown.py
# ───────────────────────────────────────────────────────────────────
# Exports merged speaker/text segments into:
# 1) a raw JSON dump, and
# 2) a human-readable Markdown transcript grouped by time blocks.

from typing import List, Dict

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
        ...

    def export_json(
        self,
        merged_segments: List[Dict[str, any]]
    ) -> None:
        """
        Save the merged segments list to JSON.

        Args:
          merged_segments – list from SegmentMerger.merge()
        """

    def export_markdown(
        self,
        merged_segments: List[Dict[str, any]],
        block_minutes: int = 1
    ) -> None:
        """
        Write a Markdown document grouping entries into minute blocks.

        Args:
          merged_segments – list from SegmentMerger.merge()
          block_minutes   – size of each time block (in minutes)
        """
        ...
