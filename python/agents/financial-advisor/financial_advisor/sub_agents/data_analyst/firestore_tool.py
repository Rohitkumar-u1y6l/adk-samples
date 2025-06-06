from google.adk.tools.function_tool import FunctionTool
import google.generativeai as genai
import pandas as pd
from google.cloud import firestore
import json
import os
from pathlib import Path
from datetime import datetime
from .genai_setup import setup_genai

class FirestoreDataManager:
    """Handles Firestore data operations"""
    def __init__(self):
        self.db = firestore.Client()
        
    async def get_transactions(self, user_id: str = None, max_rows: int = 20):
        """Fetch transactions from Firestore"""
        try:
            # Reference to transactions collection
            transactions_ref = self.db.collection('transactions')
            
            # Build query
            query = transactions_ref
            if user_id:
                query = query.where('userId', '==', user_id)
                
            # Get documents
            docs = query.limit(max_rows).stream()
            
            # Convert to list of dicts
            transactions = []
            for doc in docs:
                data = doc.to_dict()
                # Convert Firestore Timestamp to datetime if exists
                if 'dateValue' in data:
                    data['dateValue'] = data['dateValue'].strftime('%d/%m/%y')
                transactions.append(data)
                
            # Convert to DataFrame for consistency with CSV tool
            df = pd.DataFrame(transactions)
            if 'dateValue' in df.columns:
                df['dateValue'] = pd.to_datetime(df['dateValue'], format='%d/%m/%y', errors='coerce')
                df = df.sort_values('dateValue', ascending=False)
                
            return df
            
        except Exception as e:
            print(f"Error fetching from Firestore: {str(e)}")
            return None

async def firestore_qa_llm_tool(question: str, user_id: str = None, max_rows: int = 20) -> str:
    """
    Enhanced tool for analyzing Firestore transaction data using Google's Generative AI.
    Similar to csv_qa_llm_tool but uses Firestore as data source.
    """
    # Ensure GenAI is set up
    if not setup_genai():
        return "Error: Google Generative AI is not properly configured. Please check your API key."
        
    try:
        # Get available model first - this must succeed before we continue
        GENAI_MODEL = os.getenv('GENAI_MODEL', 'models/gemini-2.0-flash-001')
        print(f"Using model: {GENAI_MODEL}")
        
        # Initialize Firestore manager and get data
        firestore_manager = FirestoreDataManager()
        transactions_df = await firestore_manager.get_transactions(user_id, max_rows)
        
        if transactions_df is None or transactions_df.empty:
            return "Error: No transaction data found in Firestore"
        
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
            print(f"Attempting to create model with name: {GENAI_MODEL}")
            model = genai.GenerativeModel(GENAI_MODEL)
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
            print(f"Error with model {GENAI_MODEL}: {str(e)}")
            return f"Error analyzing transactions: {str(e)}\nPlease check your API access and permissions."
        
    except Exception as e:
        return f"Error analyzing transactions: {str(e)}\nPlease try a more specific question or rephrase your query."

# Create the ADK tool
firestore_qa_llm = FunctionTool(firestore_qa_llm_tool)
