#!/usr/bin/env python3
"""
Explainer module for providing additional insights into forecasting results
and making the reasoning process more transparent.
"""

import numpy as np
from typing import Dict, List, Any, Optional


class ForecastExplainer:
    """
    Class to provide explanations and insights for forecasting results.
    """
    
    @staticmethod
    def explain_consistency_score(probabilities: List[float]) -> Dict[str, Any]:
        """
        Provide a detailed explanation of how the consistency score is calculated
        and what it means.
        
        Args:
            probabilities: List of probability estimates from different checks
            
        Returns:
            Dictionary with explanation components
        """
        # Basic statistics
        mean = np.mean(probabilities)
        std_dev = np.std(probabilities)
        min_prob = min(probabilities)
        max_prob = max(probabilities)
        range_prob = max_prob - min_prob
        
        # Calculate variance and consistency score
        variance = np.var(probabilities)
        consistency_score = 1 - min(1, 4 * variance)
        
        # Generate explanation
        explanation = {
            "mean_probability": round(mean, 2),
            "standard_deviation": round(std_dev, 2),
            "range": round(range_prob, 2),
            "variance": round(variance, 4),
            "consistency_score": round(consistency_score, 2),
            "interpretation": "",
            "probability_distribution": probabilities
        }
        
        # Add interpretation
        if consistency_score > 0.85:
            explanation["interpretation"] = (
                "HIGH CONSISTENCY: The different reasoning methods produced very similar "
                "probability estimates. This suggests that the forecast is fairly robust "
                "across different ways of thinking about the problem. High consistency "
                "often indicates higher confidence in the forecast."
            )
        elif consistency_score > 0.7:
            explanation["interpretation"] = (
                "MEDIUM CONSISTENCY: The different reasoning methods showed moderate agreement "
                "in their probability estimates. While there is some divergence in how different "
                "approaches assess this question, there is still reasonable alignment. This level "
                "of consistency suggests moderate confidence in the forecast."
            )
        else:
            explanation["interpretation"] = (
                "LOW CONSISTENCY: The different reasoning methods produced significantly different "
                "probability estimates. This suggests high uncertainty about the forecast. "
                "When consistency is low, it often indicates that the question is complex, "
                "information is limited, or different reasoning paths lead to different conclusions. "
                "Low consistency forecasts should be treated with caution."
            )
            
        return explanation
    
    @staticmethod
    def explain_check_type(check_type: str) -> str:
        """
        Explain what a specific consistency check type means and how it works.
        
        Args:
            check_type: Type of consistency check
            
        Returns:
            String explanation of the check type
        """
        explanations = {
            "temporal_decomposition": (
                "TEMPORAL DECOMPOSITION analyzes the question by breaking it down into a sequence "
                "of events over time. It estimates the probability of each step and combines them "
                "to get the overall probability. This approach is particularly useful for questions "
                "that involve processes unfolding over time or multiple sequential events."
            ),
            "spatial_decomposition": (
                "SPATIAL DECOMPOSITION breaks the question down into its component parts or factors. "
                "It assesses each component separately and then combines them mathematically. "
                "This approach is helpful for complex questions with multiple interacting elements "
                "or when different aspects of the problem can be analyzed independently."
            ),
            "chain_of_thought": (
                "CHAIN OF THOUGHT uses an alternative reasoning path to approach the question. "
                "This provides a different perspective or mental model that might uncover "
                "insights missed in the initial analysis. Comparing different chains of thought "
                "helps identify when forecasts might be sensitive to specific reasoning patterns."
            ),
            "alternative_framing": (
                "ALTERNATIVE FRAMING recasts the question in a different form to see if that "
                "changes the probability estimate. Sometimes, subtle changes in how a question "
                "is framed can reveal biases or lead to different insights. This approach helps "
                "test whether the forecast is robust to different ways of thinking about the same problem."
            )
        }
        
        return explanations.get(check_type, "Unknown check type")
    
    @staticmethod
    def generate_visual_explanation(probabilities: List[float]) -> str:
        """
        Generate a simple ASCII visualization of the probability distribution.
        
        Args:
            probabilities: List of probability estimates
            
        Returns:
            ASCII visualization
        """
        # Create a simple ASCII histogram
        histogram = "\nProbability Distribution:\n"
        histogram += "0%   25%   50%   75%   100%\n"
        histogram += "|-----|-----|-----|-----|\n"
        
        for i, prob in enumerate(probabilities):
            position = int(prob * 40)  # Scale to 40 characters
            bar = " " * position + "▼"
            histogram += f"{bar} Check {i} ({prob:.2f})\n"
            
        return histogram
    
    @staticmethod
    def explain_divergence(probabilities: List[float], check_types: List[str]) -> str:
        """
        Explain why probabilities might diverge across different checks.
        
        Args:
            probabilities: List of probability estimates
            check_types: List of corresponding check types
            
        Returns:
            Explanation of divergence
        """
        # Simple heuristics for explaining divergence
        if max(probabilities) - min(probabilities) < 0.2:
            return "The probability estimates show relatively low divergence, suggesting consistent reasoning across methods."
        
        # Look for patterns in divergence
        explanation = "The probability estimates show notable divergence. This could be due to:"
        
        # Check if temporal and spatial decompositions differ significantly
        temp_probs = [p for p, t in zip(probabilities, check_types) if t == "temporal_decomposition"]
        spat_probs = [p for p, t in zip(probabilities, check_types) if t == "spatial_decomposition"]
        
        if temp_probs and spat_probs and abs(np.mean(temp_probs) - np.mean(spat_probs)) > 0.2:
            explanation += "\n- Different perspectives on how events unfold over time versus how components interact"
            
        # Check for extreme outliers
        mean = np.mean(probabilities)
        outliers = [(i, p) for i, p in enumerate(probabilities) if abs(p - mean) > 0.3]
        if outliers:
            explanation += "\n- Outlier estimates from specific reasoning approaches"
            for i, p in outliers:
                explanation += f"\n  - Check {i} ({check_types[i]}) with probability {p:.2f}"
        
        # General explanations
        explanation += "\n- Different assumptions being made in different reasoning paths"
        explanation += "\n- Varying interpretations of available evidence"
        explanation += "\n- Complexity of the question that reveals different aspects through different methods"
        
        return explanation


if __name__ == "__main__":
    # Simple demonstration
    explainer = ForecastExplainer()
    probs = [0.7, 0.65, 0.3, 0.75]
    check_types = ["temporal_decomposition", "spatial_decomposition", 
                  "chain_of_thought", "alternative_framing"]
    
    explanation = explainer.explain_consistency_score(probs)
    print("Consistency Score Explanation:")
    print(f"Mean probability: {explanation['mean_probability']}")
    print(f"Standard deviation: {explanation['standard_deviation']}")
    print(f"Consistency score: {explanation['consistency_score']}")
    print(f"Interpretation: {explanation['interpretation']}")
    
    print("\nCheck Type Explanations:")
    for check_type in check_types:
        print(f"{check_type}: {explainer.explain_check_type(check_type)[:100]}...")
        
    print(explainer.generate_visual_explanation(probs))
    
    print("\nDivergence Explanation:")
    print(explainer.explain_divergence(probs, check_types)) 