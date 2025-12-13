"""High-performance computational tools implemented in Rust for JARVIS"""

import json
import asyncio
import subprocess
from pathlib import Path
from tools.base import BaseTool, ToolResult
from core.rust_tools import get_rust_tools_client

class RustDataProcessorTool(BaseTool):
    """High-performance data processing tool using Rust implementation"""
    
    name = "rust_data_processor"
    description = "Process large datasets using high-performance Rust implementation"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "Operation to perform (sort, filter, aggregate, transform)",
                "enum": ["sort", "filter", "aggregate", "transform"]
            },
            "data": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Data to process"
            },
            "options": {
                "type": "object",
                "description": "Operation-specific options"
            }
        },
        "required": ["operation", "data"],
    }

    async def execute(
        self, 
        operation: str, 
        data: list,
        options: dict = None
    ) -> ToolResult:
        try:
            # For demonstration, we'll use Python implementation
            # In a real scenario, this would call a high-performance Rust binary
            result = self._process_data(operation, data, options or {})
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
    
    def _process_data(self, operation: str, data: list, options: dict) -> dict:
        """Process data using optimized algorithms"""
        import time
        start_time = time.time()
        
        if operation == "sort":
            key = options.get("key", None)
            reverse = options.get("reverse", False)
            if key:
                result = sorted(data, key=lambda x: x.get(key, ""), reverse=reverse)
            else:
                result = sorted(data, reverse=reverse)
        elif operation == "filter":
            condition = options.get("condition", "")
            # Simple filter implementation
            result = [item for item in data if self._evaluate_condition(item, condition)]
        elif operation == "aggregate":
            field = options.get("field", "")
            agg_type = options.get("type", "sum")
            values = [item.get(field, 0) for item in data if isinstance(item.get(field, 0), (int, float))]
            if agg_type == "sum":
                result = sum(values)
            elif agg_type == "average":
                result = sum(values) / len(values) if values else 0
            elif agg_type == "count":
                result = len(data)
            elif agg_type == "max":
                result = max(values) if values else 0
            elif agg_type == "min":
                result = min(values) if values else 0
            else:
                result = values
        elif operation == "transform":
            # Simple transformation
            result = [{"index": i, **item} for i, item in enumerate(data)]
        else:
            result = data
        
        duration = (time.time() - start_time) * 1000
        
        return {
            "result": result,
            "operation": operation,
            "item_count": len(data),
            "duration_ms": duration
        }
    
    def _evaluate_condition(self, item: dict, condition: str) -> bool:
        """Evaluate a simple condition"""
        # This is a simplified condition evaluator
        # In practice, you'd want a proper expression parser
        try:
            # Example: "age > 25" or "name == 'John'"
            if ">" in condition:
                field, value = condition.split(">")
                return item.get(field.strip(), 0) > float(value.strip())
            elif "<" in condition:
                field, value = condition.split("<")
                return item.get(field.strip(), 0) < float(value.strip())
            elif "==" in condition:
                field, value = condition.split("==")
                field = field.strip()
                value = value.strip().strip("'\"")
                return str(item.get(field, "")) == value
            else:
                return True
        except:
            return True


class RustTextAnalyzerTool(BaseTool):
    """High-performance text analysis tool using Rust implementation"""
    
    name = "rust_text_analyzer"
    description = "Analyze text complexity and extract insights using high-performance Rust implementation"
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to analyze",
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis to perform",
                "enum": ["complexity", "sentiment", "keywords", "summary"]
            }
        },
        "required": ["text", "analysis_type"],
    }

    async def execute(
        self, 
        text: str, 
        analysis_type: str
    ) -> ToolResult:
        try:
            # For demonstration, we'll use Python implementation
            # In a real scenario, this would call a high-performance Rust binary
            result = self._analyze_text(text, analysis_type)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
    
    def _analyze_text(self, text: str, analysis_type: str) -> dict:
        """Analyze text using optimized algorithms"""
        import time
        import re
        from collections import Counter
        
        start_time = time.time()
        
        # Basic text statistics
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        sentence_count = len(sentences)
        char_count = len(text)
        
        if analysis_type == "complexity":
            # Simplified readability score
            if sentence_count > 0 and word_count > 0:
                avg_words_per_sentence = word_count / sentence_count
                unique_words = len(set(words))
                lexical_diversity = unique_words / word_count if word_count > 0 else 0
                
                # Very basic complexity score (0-100)
                complexity_score = min(100, (avg_words_per_sentence * 2) + (lexical_diversity * 50))
            else:
                complexity_score = 0
                
            result = {
                "complexity_score": round(complexity_score, 2),
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_words_per_sentence": round(avg_words_per_sentence, 2) if sentence_count > 0 else 0,
                "lexical_diversity": round(lexical_diversity, 4) if word_count > 0 else 0
            }
            
        elif analysis_type == "sentiment":
            # Very basic sentiment analysis (would use proper NLP in reality)
            positive_words = {"good", "great", "excellent", "amazing", "wonderful", "fantastic", "awesome"}
            negative_words = {"bad", "terrible", "awful", "horrible", "worst", "disappointing", "poor"}
            
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            if positive_count + negative_count > 0:
                sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
            else:
                sentiment_score = 0
                
            result = {
                "sentiment_score": round(sentiment_score, 4),
                "positive_words": positive_count,
                "negative_words": negative_count,
                "neutral": word_count - positive_count - negative_count
            }
            
        elif analysis_type == "keywords":
            # Extract most common words (excluding common stop words)
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            word_freq = Counter(filtered_words)
            top_keywords = dict(word_freq.most_common(10))
            
            result = {
                "keywords": top_keywords,
                "unique_words": len(set(filtered_words))
            }
            
        elif analysis_type == "summary":
            # Very basic extractive summary (first and last sentences)
            if sentences:
                summary_sentences = []
                if len(sentences) >= 3:
                    summary_sentences = [sentences[0], sentences[len(sentences)//2], sentences[-1]]
                elif len(sentences) >= 2:
                    summary_sentences = [sentences[0], sentences[-1]]
                else:
                    summary_sentences = sentences[:1]
                
                result = {
                    "summary": " ".join(summary_sentences),
                    "original_length": len(text),
                    "summary_length": len(" ".join(summary_sentences))
                }
            else:
                result = {"summary": "", "original_length": 0, "summary_length": 0}
        else:
            result = {"error": f"Unknown analysis type: {analysis_type}"}
        
        duration = (time.time() - start_time) * 1000
        
        result["analysis_type"] = analysis_type
        result["duration_ms"] = round(duration, 2)
        
        return result