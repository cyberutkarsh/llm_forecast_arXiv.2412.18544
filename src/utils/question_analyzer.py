#!/usr/bin/env python3
"""
Utility for analyzing forecasting questions using LLM reasoning.
"""

import openai
import json
import os
from typing import Dict, Any, Optional

class QuestionAnalyzer:
    """Uses LLM to analyze forecasting questions."""

    def __init__(self, model="gpt-4o"):
        """Initialize the question analyzer."""
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze a forecasting question to extract key information.
        
        Args:
            question: The forecasting question
            
        Returns:
            Dictionary containing extracted information
        """
        prompt = f"""
        Analyze this forecasting question: "{question}"
        
        Extract key information like:
        1. What entity (person, company, product, etc.) is involved
        2. Any specific numeric thresholds mentioned
        3. Any time frames mentioned
        4. The nature of the prediction (will X happen, will X exceed Y, etc.)
        
        Respond with valid JSON that includes these fields (if present in the question):
        - question_type: forecasting domain (e.g., "financial", "political", "weather", "sports", etc.)
        - entity: the main subject of the question
        - identifier: any identifier for the entity (e.g., ticker symbol for stocks)
        - threshold: any numeric threshold mentioned (as a number, not string)
        - threshold_unit: the unit of the threshold (e.g., "dollars", "percent", "points")
        - direction: relationship to threshold (e.g., "above", "below", "exactly")
        - timeframe: when the prediction is for (e.g., "next month", "by end of 2024")
        - event_description: brief description of the event being predicted
        
        Include only fields that are explicitly or strongly implied in the question.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            # Add a default field to indicate if analysis was successful
            result["analysis_success"] = True
            return result
        except Exception as e:
            print(f"Error analyzing question: {e}")
            return {"analysis_success": False, "error": str(e)}
    
    def get_search_query(self, analysis: Dict[str, Any]) -> Optional[str]:
        """
        Generate an optimized search query based on question analysis.
        
        Args:
            analysis: The question analysis from analyze_question
            
        Returns:
            Optimized search query or None if analysis unsuccessful
        """
        if not analysis.get("analysis_success", False):
            return None
            
        # Extract all available fields
        question_type = analysis.get("question_type", "")
        entity = analysis.get("entity", "")
        identifier = analysis.get("identifier", "")
        threshold = analysis.get("threshold", "")
        threshold_unit = analysis.get("threshold_unit", "")
        direction = analysis.get("direction", "")
        timeframe = analysis.get("timeframe", "")
        event_description = analysis.get("event_description", "")
        
        # For financial price queries, directly construct a clean query without thresholds
        if question_type.lower() == "financial" and threshold and threshold_unit:
            # Create a clean price-focused query without mentioning the threshold
            if entity:
                # Include identifier (like ticker symbol) if available
                if identifier and identifier.lower() != entity.lower():
                    price_query = f"{entity} {identifier} current price real-time"
                else:
                    price_query = f"{entity} current price real-time"
                print(f"[Analysis] Using direct price query: '{price_query}'")
                return price_query
        
        # Use LLM to generate an appropriate search query for other question types
        prompt = f"""
        Create an optimized web search query for finding relevant, factual information about the following forecasting question.
        
        Information about the question:
        - Domain/Type: {question_type}
        - Main entity/subject: {entity}
        - Identifier (if applicable): {identifier}
        - Numeric threshold (if any): {threshold} {threshold_unit}
        - Direction relative to threshold: {direction}
        - Time frame: {timeframe}
        - Event being predicted: {event_description}
        
        Guidelines for creating the search query:
        1. Focus on current factual information rather than predictions
        2. For financial questions about prices:
           - IMPORTANT: DO NOT include the threshold value in the search query
           - Use a simple format like "[ENTITY NAME] [IDENTIFIER] current price real-time"
        3. For time references:
           - If a specific year/date is mentioned in the question, include it
           - If only relative time is mentioned (like "tomorrow", "next week"), DO NOT add a specific year
        4. For financial questions:
           - For assets, stocks, cryptocurrencies, or commodities, focus on "current price" and "real-time" data
           - For economic metrics, focus on recent trends and specific numbers
        5. For political questions, focus on key entities and current positions
        6. For sports questions, focus on team statistics and recent performance
        7. For weather questions, focus on current conditions and forecasts
        8. Keep the query concise (5-10 words) and focused on the most important elements
        9. Avoid including terms like "prediction", "forecast", or "will" as these tend to return speculation
        10. The query should look like a typical search engine query, not a complete sentence
        11. IMPORTANT: Do NOT add years or dates that weren't in the original question
        12. For any questions involving numeric values or thresholds, prioritize getting current real-time data
        
        Return ONLY the search query text without any explanation, quotes, or additional context.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract and clean the generated search query
            search_query = response.choices[0].message.content.strip()
            
            # Remove any quotation marks or markdown formatting
            search_query = search_query.replace('"', '').replace('`', '').strip()
            
            print(f"[Search] Generated optimized query: '{search_query}'")
            return search_query
            
        except Exception as e:
            print(f"Error generating search query: {e}")
            
            # Fall back to a simple query if LLM generation fails
            if question_type.lower() == "financial" and entity:
                fallback_query = f"{entity} current price"
            else:
                fallback_query = f"{entity} {event_description} {timeframe}"
            print(f"[Search] Using fallback query: '{fallback_query}'")
            return fallback_query

if __name__ == "__main__":
    # Test the analyzer
    analyzer = QuestionAnalyzer()
    
    test_questions = [
        "Will AMD stock cross 110 dollars next month?",
        "Is NVDA going to hit $200 this year?",
        "Will inflation exceed 3% in the next quarter?",
        "Will Ukraine and Russia reach a peace deal in 2024?",
        "Will the Patriots win more than 8 games next season?"
    ]
    
    for question in test_questions:
        print(f"\nAnalyzing: {question}")
        analysis = analyzer.analyze_question(question)
        print(f"Analysis: {json.dumps(analysis, indent=2)}")
        
        search_query = analyzer.get_search_query(analysis)
        print(f"Search query: {search_query}") 