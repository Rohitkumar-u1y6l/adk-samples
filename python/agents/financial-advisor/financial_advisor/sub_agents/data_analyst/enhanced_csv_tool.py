from google.adk.tools.function_tool import FunctionTool
from google.adk import Agent, InvocationContext
from google.adk.tools.tool_context import ToolContext
import pandas as pd
from typing import Dict, Any
import json
from datetime import datetime

class TransactionAnalyzer:
    """Helper class for sophisticated financial transaction analysis"""
    
    TRANSACTION_CATEGORIES = {
        'PAYMENT': ['upi', 'pay', 'sent', 'transfer', 'trf'],
        'BILL': ['bill', 'utility', 'electricity', 'water', 'gas', 'internet'],
        'INCOME': ['salary', 'interest', 'credit', 'received', 'refund'],
        'SHOPPING': ['purchase', 'shopping', 'mart', 'store', 'shop'],
        'ENTERTAINMENT': ['movie', 'restaurant', 'dining', 'cafe', 'food'],
        'INVESTMENT': ['mutual fund', 'stocks', 'shares', 'investment', 'dividend'],
        'WITHDRAWAL': ['atm', 'withdrawal', 'cash'],
        'SUBSCRIPTION': ['subscription', 'netflix', 'amazon prime', 'spotify'],
        'EMI': ['emi', 'loan', 'mortgage', 'installment'],
        'TRAVEL': ['travel', 'flight', 'hotel', 'bus', 'train', 'taxi', 'uber']
    }
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with transaction DataFrame and perform preprocessing"""
        self.df = df.copy()  # Create copy to prevent modifying original
        self.validate_data()
        self._preprocess_data()
        self._calculate_statistics()
        
    def validate_data(self):
        """Validate required columns and data types"""
        required_columns = ['dateValue', 'mentionText', 'amount']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
    def _preprocess_data(self):
        """Enhanced preprocessing with better date handling and amount normalization"""
        try:
            # Date processing with better error handling
            self.df['parsed_date'] = pd.to_datetime(self.df['dateValue'], 
                                                  format='%d/%m/%y', 
                                                  errors='coerce')
            
            # Mark records with invalid dates
            self.df['has_valid_date'] = ~self.df['parsed_date'].isna()
            
            # Enhanced transaction classification
            self.df['transaction_type'] = self.df['mentionText'].apply(self._classify_transaction)
            self.df['transaction_category'] = self.df['mentionText'].apply(self._categorize_transaction)
            
            # Add derived time-based features
            self.df['month_year'] = self.df['parsed_date'].dt.to_period('M')
            self.df['day_of_week'] = self.df['parsed_date'].dt.day_name()
            self.df['is_weekend'] = self.df['parsed_date'].dt.dayofweek.isin([5, 6])
            
            # Convert amount to numeric, handling any non-numeric values
            self.df['amount'] = pd.to_numeric(self.df['amount'], errors='coerce')
            
            # Add basic transaction attributes
            self.df['is_credit'] = self.df['amount'] > 0
            self.df['is_high_value'] = self.df['amount'].abs() > self.df['amount'].abs().quantile(0.9)
            
        except Exception as e:
            raise ValueError(f"Error during preprocessing: {str(e)}")
            
    def _calculate_statistics(self):
        """Calculate and store useful statistical information"""
        self.statistics = {
            'total_transactions': len(self.df),
            'date_range': {
                'start': self.df['parsed_date'].min().strftime('%d/%m/%Y'),
                'end': self.df['parsed_date'].max().strftime('%d/%m/%Y')
            },
            'total_credits': self.df[self.df['amount'] > 0]['amount'].sum(),
            'total_debits': self.df[self.df['amount'] < 0]['amount'].sum(),
            'transaction_types': self.df['transaction_type'].value_counts().to_dict(),
            'categories': self.df['transaction_category'].value_counts().to_dict(),
            'monthly_averages': self.df.groupby('month_year')['amount'].agg(['count', 'sum', 'mean']).to_dict(),
            'weekend_vs_weekday': {
                'weekend_avg': self.df[self.df['is_weekend']]['amount'].mean(),
                'weekday_avg': self.df[~self.df['is_weekend']]['amount'].mean()
            }
        }
            
    def _classify_transaction(self, text: str) -> str:
        """Enhanced transaction type classification"""
        if pd.isna(text):
            return 'UNKNOWN'
        
        text = text.lower()
        
        # UPI-specific classification
        if 'upi' in text:
            if any(word in text for word in ['received', 'credit', 'credited']):
                return 'UPI_RECEIVED'
            return 'UPI_PAYMENT'
            
        # Other transaction types
        type_indicators = {
            'BILL_PAYMENT': ['billpay', 'bill payment', 'utility'],
            'SALARY': ['salary', 'sal cr', 'monthly pay'],
            'INTEREST': ['interest', 'int.', 'int cr'],
            'ATM': ['atm', 'cash withdrawal'],
            'TRANSFER': ['transfer', 'trf', 'neft', 'rtgs', 'imps'],
            'INVESTMENT': ['investment', 'mutual fund', 'shares'],
            'REFUND': ['refund', 'cashback', 'return'],
            'LOAN': ['loan', 'emi', 'mortgage']
        }
        
        for tx_type, indicators in type_indicators.items():
            if any(ind in text for ind in indicators):
                return tx_type
                
        return 'OTHER'
        
    def _categorize_transaction(self, text: str) -> str:
        """Categorize transaction into predefined categories"""
        if pd.isna(text):
            return 'UNKNOWN'
            
        text = text.lower()
        
        for category, keywords in self.TRANSACTION_CATEGORIES.items():
            if any(keyword in text for keyword in keywords):
                return category
                
        return 'OTHER'
        
    def get_context_for_llm(self, question: str) -> Dict[str, Any]:
        """Prepare comprehensive context for LLM analysis"""
        try:
            # Start with basic statistics
            context = {
                'statistics': self.statistics,
                'question': question
            }
            
            # Add question-specific analysis
            sample = self.df
            
            # Time-based filtering
            if 'recent' in question.lower():
                sample = self.df.sort_values('parsed_date', ascending=False)
            elif 'last month' in question.lower():
                last_month = self.df['month_year'].max()
                sample = self.df[self.df['month_year'] == last_month]
                
            # Category-based filtering
            for category in self.TRANSACTION_CATEGORIES:
                if category.lower() in question.lower():
                    sample = self.df[self.df['transaction_category'] == category]
                    
            # Value-based analysis
            if 'highest' in question.lower() or 'top' in question.lower():
                sample = sample.nlargest(5, 'amount')
            elif 'lowest' in question.lower():
                sample = sample.nsmallest(5, 'amount')
                
            # Pattern detection
            if 'pattern' in question.lower() or 'trend' in question.lower():
                context['patterns'] = {
                    'monthly_trend': self.df.groupby('month_year')['amount'].sum().to_dict(),
                    'category_distribution': self.df.groupby('transaction_category')['amount'].agg(['count', 'sum']).to_dict(),
                    'day_of_week_pattern': self.df.groupby('day_of_week')['amount'].mean().to_dict()
                }
                
            # Add relevant sample data
            context['relevant_transactions'] = sample.head(10).to_dict(orient='records')
            
            return context
            
        except Exception as e:
            return {
                'error': str(e),
                'question': question
            }

async def csv_qa_llm_tool(question: str) -> str:
    """
    Enhanced ADK Tool for sophisticated financial transaction analysis.
    Uses TransactionAnalyzer for comprehensive data processing and LLM for intelligent insights.
    """
    try:
        # Load and analyze data with error handling
        try:
            df = pd.read_csv('/workspaces/adk-samples/data/transactions.csv')
        except Exception as e:
            return f"Error loading transaction data: {str(e)}"

        # Initialize analyzer with validation
        try:
            analyzer = TransactionAnalyzer(df)
        except ValueError as e:
            return f"Error in data validation: {str(e)}"

        # Get enriched context for LLM
        context = analyzer.get_context_for_llm(question)
        
        if 'error' in context:
            return f"Error analyzing data: {context['error']}"

        # Create a sophisticated prompt for the LLM
        prompt = f"""You are an expert financial data analyst with these advanced capabilities:
- Deep understanding of transaction patterns and user financial behavior
- Sophisticated trend analysis and anomaly detection
- Strategic financial insights and personalized recommendations
- Clear explanation of complex financial patterns

Analyze this comprehensive transaction data context:
```json
{json.dumps(context, default=str, indent=2)}
```

Question: "{question}"

Provide a detailed analysis following this structure:

1. 📊 Direct Answer
   - Clear, specific response to the question
   - Include relevant numbers and statistics
   - Highlight key findings

2. 💡 Key Insights
   - Notable patterns or trends
   - Unusual activities or anomalies
   - Comparative analysis (e.g., month-over-month, weekday vs weekend)

3. ⚠️ Important Considerations
   - Data limitations or caveats
   - Missing or incomplete information
   - Potential areas needing attention

4. 📈 Actionable Recommendations
   - Specific suggestions based on the analysis
   - Steps to improve financial patterns
   - Areas for potential optimization

Use clear formatting with bullet points and sections. Include specific numbers and percentages where relevant.
Explain any financial terms in simple language.
"""

        # Create a context-aware tool call
        tool_context = ToolContext(model="gemini-2.5-pro-preview-05-06")
        
        try:
            response = await Agent.generate_text_async(
                prompt,
                temperature=0.1,  # Low temperature for more focused analysis
                tool_context=tool_context
            )
            return response
        except Exception as e:
            return f"Error generating analysis: {str(e)}"
            
    except Exception as e:
        return f"Unexpected error in analysis process: {str(e)}"

# Create the ADK tool
csv_qa_llm = FunctionTool(csv_qa_llm_tool)
