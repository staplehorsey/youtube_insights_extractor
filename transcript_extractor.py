import os
import requests
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from typing import Optional
import json
from datetime import datetime

@dataclass
class VideoInsights:
    running_notes: str
    ai_tools_json: str
    final_summary: Optional[str] = None
    timestamp_map: Optional[Dict[int, str]] = None

@dataclass
class ProviderPreferences:
    sort: Optional[str] = None  # "throughput" or "price"
    order: Optional[List[str]] = None  # List of provider names in priority order
    allow_fallbacks: bool = True

class TranscriptInsightExtractor:
    def __init__(self, 
                 openrouter_api_key: str, 
                 overlap_tokens: int = 1000,
                 base_model: str = "google/gemini-2.0-flash-exp:free",
                 thinking_model: str = "google/gemini-2.0-flash-thinking-exp:free",
                 base_model_max_tokens: int = 4000,
                 thinking_model_max_tokens: int = 4000,
                 max_output_tokens: Optional[int] = None,
                 provider_preferences: Optional[ProviderPreferences] = None):
        self.api_key = openrouter_api_key
        self.overlap_tokens = overlap_tokens
        self.base_model = base_model  # For straightforward tasks like note-taking
        self.thinking_model = thinking_model  # For complex analysis and summaries
        self.base_model_max_tokens = base_model_max_tokens
        self.thinking_model_max_tokens = thinking_model_max_tokens
        self.max_output_tokens = max_output_tokens
        self.provider_preferences = provider_preferences
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def _call_llm(self, prompt: str, model: str) -> str:
        """Make API call to OpenRouter"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        # Only set max_tokens if specified in output_tokens
        if self.max_output_tokens is not None:
            data["max_tokens"] = self.max_output_tokens
            
        if self.provider_preferences:
            provider_config = {}
            if self.provider_preferences.sort:
                provider_config["sort"] = self.provider_preferences.sort
            if self.provider_preferences.order:
                provider_config["order"] = self.provider_preferences.order
            if not self.provider_preferences.allow_fallbacks:
                provider_config["allow_fallbacks"] = False
            if provider_config:
                data["provider"] = provider_config
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            if response.status_code != 200:
                print(f"Error response from API: {response.text}")
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"API call failed: {str(e)}")
            print(f"Request URL: {self.api_url}")
            print(f"Request headers: {headers}")
            print(f"Request data: {data}")
            raise

    def _create_note_taking_prompt(self, text: str) -> str:
        return f"""Take detailed notes on this transcript section, paying attention to timestamps:
{text}

Focus on:
1. Key points and topics discussed, including when they occur (use timestamp ranges)
2. Any AI tools mentioned and their context, noting the timestamp ranges where they are discussed
3. Important quotes or examples

Format your notes with timestamp ranges like this:
[00:30-01:45] Note content here
[01:45-02:30] Another note here

For AI tools specifically, make sure to clearly indicate the full timestamp range where each tool is discussed."""

    def _create_tools_analysis_prompt(self, notes: str) -> str:
        return f"""Analyze these timestamped notes from a video and extract information about all AI tools mentioned:

{notes}

For each AI tool, find the timestamp ranges where it is discussed in the notes (format: [MM:SS-MM:SS]).

ONLY OUTPUT VALID JSON. DO NOT OUTPUT ANYTHING ELSE. NO EXPLANATIONS.

Format your response as JSON:
{{
    "ai_tools": [
        {{
            "name": "tool name",
            "description": "brief description of the tool",
            "timestamp_ranges": ["00:30-01:45", "02:15-03:00"],  # All ranges where this tool is discussed
            "usage_context": "how it was used or discussed",
            "sentiment": "positive/negative/mixed",
            "features": ["feature 1", "feature 2"],
            "limitations": ["limitation 1", "limitation 2"],
            "use_cases": ["use case 1", "use case 2"],
            "integrations": ["integration 1", "integration 2"],
            "pricing": "pricing information if mentioned or null",
            "examples": ["example 1", "example 2"]
        }}
    ]
}}"""

    def _create_final_summary_prompt(self, notes: str) -> str:
        return f"""Create a high-level executive summary of this video based on these notes:

{notes}

Focus on:
1. The main purpose and key message of the video
2. The most significant insights or takeaways
3. Who would benefit most from this content
4. Any overarching themes or patterns

Keep the summary concise but comprehensive, focusing on the big picture rather than specific details."""

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count based on words"""
        return len(text.split()) * 1.3  # Rough estimate: 1 word ≈ 1.3 tokens

    def _chunk_transcript(self, formatted_segments: List[str], max_model_tokens: int) -> List[List[str]]:
        """Split transcript into chunks that fit within token limits with overlap"""
        chunks = []
        current_chunk = []
        current_tokens = 0
        total_tokens = 0
        last_chunk_start_idx = 0
        
        # Reserve tokens for prompt and overhead
        available_tokens = max_model_tokens - 1000  # Reserve 1000 tokens for prompt
        
        for i, segment in enumerate(formatted_segments):
            segment_tokens = self._estimate_tokens(segment)
            
            if current_tokens + segment_tokens > available_tokens:
                # Add current chunk
                chunks.append(current_chunk)
                
                # Calculate overlap start index
                overlap_start = max(last_chunk_start_idx + 1, i - self.overlap_tokens)
                
                # Start new chunk with overlap
                current_chunk = formatted_segments[overlap_start:i]
                current_tokens = sum(self._estimate_tokens(s) for s in current_chunk)
                last_chunk_start_idx = overlap_start
            
            current_chunk.append(segment)
            current_tokens += segment_tokens
            total_tokens += segment_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        print(f"Total tokens: {total_tokens}")    
        return chunks

    def _merge_running_notes(self, notes_chunks: List[str]) -> str:
        """Merge multiple note chunks while removing potential duplicates and overlaps"""
        if not notes_chunks:
            return ""
            
        # Simple concatenation for now - could be enhanced with more sophisticated merging
        merged_notes = "\n\n".join(notes_chunks)
        return merged_notes

    def process_transcript(self, transcript_data: List[Dict[str, Any]]) -> VideoInsights:
        # Format transcript with embedded timestamps
        formatted_segments = []
        for segment in transcript_data:
            start_time = int(segment['start'])
            minutes = start_time // 60
            seconds = start_time % 60
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            formatted_segments.append(f"{timestamp} {segment['text']}")
        
        # Split into chunks with overlap, using base model tokens since we use it for note taking
        transcript_chunks = self._chunk_transcript(formatted_segments, self.base_model_max_tokens)
        running_notes_chunks = []
        
        # Process each chunk
        for i, chunk in enumerate(transcript_chunks):
            chunk_text = ' '.join(chunk)
            print(f"Processing chunk {i+1}/{len(transcript_chunks)}...")
            
            chunk_notes = self._call_llm(
                prompt=self._create_note_taking_prompt(chunk_text),
                model=self.base_model,
            )
            running_notes_chunks.append(chunk_notes)
        
        # Merge all notes
        running_notes = self._merge_running_notes(running_notes_chunks)
        
        # Process tools and create summary
        print("Analyzing AI tools...")
        tools_prompt = self._create_tools_analysis_prompt(running_notes)
        ai_tools_json = self._call_llm(
            prompt=tools_prompt,
            model=self.base_model,
        )
        
        # Create final summary
        print("Creating final summary...")
        summary_prompt = self._create_final_summary_prompt(running_notes)
        final_summary = self._call_llm(
            prompt=summary_prompt,
            model=self.thinking_model,
        )
        
        return VideoInsights(
            running_notes=running_notes,
            ai_tools_json=ai_tools_json,
            final_summary=final_summary
        )

    def format_insights_markdown(self, insights: VideoInsights, video_id: str = 'video_id') -> str:
        """Format insights as a markdown document"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Parse AI tools JSON and format with timestamp links
        try:
            tools_data = json.loads(insights.ai_tools_json)
            formatted_tools = []
            
            for tool in tools_data['ai_tools']:
                tool_name = tool['name']
                description = tool['description']
                timestamp_ranges = tool.get('timestamp_ranges', [])
                
                formatted_tool = f"### {tool_name}\n"
                
                # Add timestamp links for each range
                if timestamp_ranges:
                    links = []
                    for time_range in timestamp_ranges:
                        start_time = time_range.split('-')[0].strip()  # Format: "MM:SS"
                        minutes, seconds = map(int, start_time.split(':'))
                        total_seconds = minutes * 60 + seconds
                        # Fix URL format to properly include video ID and timestamp
                        video_url = f"https://www.youtube.com/watch?v={video_id}&t={total_seconds}s"
                        links.append(f"[🎥 {time_range}]({video_url})")
                    
                    formatted_tool += f"**Discussed at:** {' | '.join(links)}\n\n"
                
                formatted_tool += f"{description}\n"
                
                if tool.get('usage_context'):
                    formatted_tool += f"\n**Usage Context:** {tool['usage_context']}\n"
                if tool.get('sentiment'):
                    formatted_tool += f"\n**Sentiment:** {tool['sentiment']}\n"
                if tool.get('features'):
                    formatted_tool += "\n**Features:**\n" + "\n".join(f"- {f}" for f in tool['features']) + "\n"
                if tool.get('limitations'):
                    formatted_tool += "\n**Limitations:**\n" + "\n".join(f"- {l}" for l in tool['limitations']) + "\n"
                if tool.get('use_cases'):
                    formatted_tool += "\n**Use Cases:**\n" + "\n".join(f"- {u}" for u in tool['use_cases']) + "\n"
                if tool.get('integrations'):
                    formatted_tool += "\n**Integrations:**\n" + "\n".join(f"- {i}" for i in tool['integrations']) + "\n"
                if tool.get('pricing') and tool['pricing'] != "null":
                    formatted_tool += f"\n**Pricing:** {tool['pricing']}\n"
                if tool.get('examples'):
                    formatted_tool += "\n**Examples:**\n" + "\n".join(f"- {e}" for e in tool['examples']) + "\n"
                
                formatted_tools.append(formatted_tool)
            
            tools_section = "\n\n".join(formatted_tools)
        except json.JSONDecodeError as e:
            print("Failed to decode JSON for tools", e)
            tools_section = f"```\n{insights.ai_tools_json}\n```"
        
        markdown = f"""# Video Analysis Report
Generated on: {current_time}
Video ID: {video_id}

## Executive Summary
{insights.final_summary}

## AI Tools Mentioned
{tools_section}

## Detailed Notes
{insights.running_notes}
"""
        return markdown

    def analyze_youtube_video(self, video_id: str) -> str:
        """
        Analyze a YouTube video by its ID and return formatted markdown insights.
        
        Args:
            video_id (str): The YouTube video ID (e.g., "RXZPxl7uEQQ")
            
        Returns:
            str: Formatted markdown with video insights
            
        Example:
            ```python
            from IPython.display import display, Markdown
            from youtube_transcript_api import YouTubeTranscriptApi
            
            extractor = TranscriptInsightExtractor(openrouter_api_key="your-key")
            markdown_output = extractor.analyze_youtube_video("RXZPxl7uEQQ")
            display(Markdown(markdown_output))
            ```
        """
        try:
            # Import here to avoid making it a required dependency
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError(
                "youtube_transcript_api is required for this function. "
                "Install it with: pip install youtube_transcript_api"
            )
            
        print(f"Fetching transcript for video {video_id}...")
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        print("Processing transcript...")
        insights = self.process_transcript(transcript)
        
        print("Formatting results...")
        return self.format_insights_markdown(insights, video_id)
