from google.adk.tools.function_tool import FunctionTool
import google.generativeai as genai
import pandas as pd
import json
import os
from pathlib import Path
from .genai_setup import setup_genai

async def get_available_model():
    """Get basic model that's guaranteed to work"""
    try:
        # Define available Vertex AI models in order of preference
        VERTEX_MODELS = [
            "gemini-2.0-flash-001",  # Gemini Flash model
            "gemini-pro",
            "gemini-1.0-pro",
            "text-bison@002",
            "text-bison@001"
        ]
        
        # Allow override through environment variable
        GENAI_MODEL = os.getenv('GENAI_MODEL', VERTEX_MODELS[0])
        if GENAI_MODEL:
            print(f"Using model from environment: {GENAI_MODEL}")
            return f"models/{GENAI_MODEL}"
            
        # Get list of available models
        models = genai.list_models()
        if not models:
            print("Warning: No models returned from API")
            print("Please check Vertex AI Model Garden: https://console.cloud.google.com/vertex-ai/model-garden")
            return None
            
        # Print all models for debugging
        print("All available models:", [m.name for m in models])
            
        # First try: Look for models that list generateContent in their methods
        available_models = [m for m in models 
                          if hasattr(m, 'supported_generation_methods') 
                          and 'generateContent' in m.supported_generation_methods]
        
        if not available_models:
            # Second try: Look for any text or chat model
            available_models = [m for m in models 
                              if any(x in m.name.lower() for x in ['text', 'chat', 'gemini'])]
            
        if not available_models:
            # Last resort: Try all models
            available_models = models
            
        if not available_models:
            print("Warning: No models found at all")
            return None
            
        # Log available models
        print(f"Available models: {[m.name for m in available_models]}")
        
        # Use the first available model (simplest approach)
        selected_model = available_models[0].name
        print(f"Selected model: {selected_model}")
        return selected_model
        
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return None

async def csv_qa_llm_tool(question: str, max_rows: int = 20) -> str:
    """
    Enhanced tool for analyzing CSV transaction data using Google's Generative AI.
    """
    # Ensure GenAI is set up
    if not setup_genai():
        return "Error: Google Generative AI is not properly configured. Please check your API key."
        
    try:
        # Get available model first - this must succeed before we continue
        model_name = await get_available_model()
        if not model_name:
            return "Error: No suitable model found. Please check your API access and permissions. Check logs for available models."
            
        # Model is now confirmed available
        print(f"Proceeding with model: {model_name}")
        
        # Load data
        data_path = Path('/workspaces/adk-samples/data/transactions.csv')
        if not data_path.exists():
            return "Error: Transaction data file not found"
            
        try:
            transactions_df = pd.read_csv(data_path)
            # Sort by date if available
            if 'dateValue' in transactions_df.columns:
                transactions_df['dateValue'] = pd.to_datetime(transactions_df['dateValue'], format='%d/%m/%y', errors='coerce')
                transactions_df = transactions_df.sort_values('dateValue', ascending=False)
        except Exception as e:
            return f"Error loading transaction data: {str(e)}"
        
        # Get subset based on question
        if 'highest' in question.lower() or 'top' in question.lower():
            if 'amount' in transactions_df.columns:
                transactions_df = transactions_df.nlargest(5, 'amount')
        
        # Create a subset of the data for analysis
        sample = transactions_df.head(max_rows).fillna('')
        transactions_text = sample.to_string(index=False)
        
        # Create a structured prompt for the LLM
        prompt = f"""You are a helpful financial analysis assistant. Answer this question about the transaction data:
"{question}"

Here is a sample of the transactions to analyze (limited to {max_rows} rows for brevity):

{transactions_text}

Columns explained:
- dateValue: Date of transaction
- mentionText: Transaction description
- amount: Transaction amount (positive for credits, negative for debits)
- Other columns: Additional transaction details

Please provide:
1. Direct answer to the question with specific numbers
2. Any relevant patterns or insights
3. Important notes about the transactions shown

Format your response with clear sections and bullet points."""

        # Create model and generate response
        try:
            print(f"Attempting to create model with name: {model_name}")
            model = genai.GenerativeModel(model_name)
            print("Model created successfully, generating content...")
            
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
            }
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            print("Content generated successfully")
            
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'parts'):
                return response.parts[0].text
            else:
                return str(response)
        except Exception as e:
            print(f"Error with model {model_name}: {str(e)}")
            return f"Error analyzing transactions: {str(e)}\nPlease check your API access and permissions."
        
    except Exception as e:
        return f"Error analyzing transactions: {str(e)}\nPlease try a more specific question or rephrase your query."

# Create the ADK tool
csv_qa_llm = FunctionTool(csv_qa_llm_tool)
