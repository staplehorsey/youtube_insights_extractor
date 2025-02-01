import os
import requests
from typing import List, Dict, Any
from dataclasses import dataclass
from typing import Optional
import json
from datetime import datetime

@dataclass
class VideoInsights:
    running_notes: str
    ai_tools_json: str
    final_summary: Optional[str] = None

class TranscriptInsightExtractor:
    def __init__(self, openrouter_api_key: str, max_tokens: int = 16000, overlap_tokens: int = 1000):
        self.api_key = openrouter_api_key
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.note_taking_model = "google/gemini-2.0-flash-exp:free"
        self.summary_llm_model = "google/gemini-2.0-flash-thinking-exp:free"
    
    def _call_llm(self, prompt: str, model: str) -> str:
        """Make API call to OpenRouter"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
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

    def _create_note_taking_prompt(self, text: str, previous_notes: Optional[str] = None) -> str:
        if previous_notes:
            context = f"""Previous notes:
{previous_notes}

Continue taking notes on this new section:
{text}

Focus on:
1. Key points and topics discussed
2. Any AI tools mentioned and their context
3. Important quotes or examples

Format as a running list of notes."""
        else:
            context = f"""Take detailed notes on this transcript section:
{text}

Focus on:
1. Key points and topics discussed
2. Any AI tools mentioned and their context
3. Important quotes or examples

Format as a running list of notes."""
        
        return context

    def _create_tools_analysis_prompt(self, notes: str) -> str:
        return f"""Analyze these notes from a video and extract information about all AI tools mentioned:

{notes}

Provide a detailed analysis of each AI tool mentioned, including:
- How the tool was used or discussed
- The sentiment around the tool (positive/negative/mixed)
- Notable features, limitations, or use cases mentioned
- Any specific examples or demonstrations
- Integration points with other tools
- Pricing information if mentioned

Format your response as JSON:
{{
    "ai_tools": [
        {{
            "name": "tool name",
            "description": "brief description of the tool",
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

    def process_transcript(self, transcript_data: List[Dict[str, Any]]) -> VideoInsights:
        # Convert transcript to text and split into chunks
        full_text = ' '.join(entry['text'] for entry in transcript_data)
        words = full_text.split()
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        # Rough token estimation
        for word in words:
            word_tokens = len(word.split()) * 1.3
            if current_tokens + word_tokens > self.max_tokens - self.overlap_tokens:
                chunks.append(' '.join(current_chunk))
                overlap_words = current_chunk[-int(self.overlap_tokens/1.3):]
                current_chunk = overlap_words + [word]
                current_tokens = len(' '.join(current_chunk).split()) * 1.3
            else:
                current_chunk.append(word)
                current_tokens += word_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Process chunks with note-taking model
        running_notes = ""
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            prompt = self._create_note_taking_prompt(chunk, running_notes if running_notes else None)
            chunk_notes = self._call_llm(
                prompt=prompt,
                model=self.note_taking_model
            )
            running_notes = f"{running_notes}\n\n{chunk_notes}" if running_notes else chunk_notes
        
        # Extract AI tools information
        print("Analyzing AI tools...")
        ai_tools_json = self._call_llm(
            prompt=self._create_tools_analysis_prompt(running_notes),
            model=self.note_taking_model
        )
        
        # Generate final summary
        print("Generating final summary...")
        final_summary = self._call_llm(
            prompt=self._create_final_summary_prompt(running_notes),
            model=self.summary_llm_model
        )
        
        return VideoInsights(
            running_notes=running_notes,
            ai_tools_json=ai_tools_json,
            final_summary=final_summary
        )

    def format_insights_markdown(self, insights: VideoInsights, video_id: str = 'video_id') -> str:
        """Format insights as a markdown document"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        markdown = f"""# Video Analysis Report
Generated on: {current_time}
Video ID: {video_id}

## Executive Summary
{insights.final_summary}

## AI Tools Mentioned
```json
{insights.ai_tools_json}
```

## Detailed Notes
{insights.running_notes}
"""
        return markdown
