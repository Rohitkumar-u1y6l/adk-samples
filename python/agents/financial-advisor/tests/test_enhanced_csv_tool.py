import pytest
import pandas as pd
import asyncio
from financial_advisor.sub_agents.data_analyst.enhanced_csv_tool import TransactionAnalyzer, csv_qa_llm_tool

# Sample test data
@pytest.fixture
def sample_transactions():
    return pd.DataFrame({
        'dateValue': ['01/06/25', '02/06/25', '03/06/25', '04/06/25'],
        'mentionText': [
            'UPI payment to grocery store',
            'Salary credit',
            'Netflix subscription',
            'ATM withdrawal'
        ],
        'amount': [-500, 5000, -199, -1000]
    })

@pytest.fixture
def analyzer(sample_transactions):
    return TransactionAnalyzer(sample_transactions)

def test_transaction_analyzer_init(analyzer):
    """Test basic initialization and preprocessing"""
    assert len(analyzer.df) == 4
    assert 'transaction_type' in analyzer.df.columns
    assert 'transaction_category' in analyzer.df.columns
    assert 'parsed_date' in analyzer.df.columns

def test_transaction_classification(analyzer):
    """Test transaction classification logic"""
    types = analyzer.df['transaction_type'].unique()
    assert 'UPI_PAYMENT' in types
    assert 'SALARY' in types

def test_transaction_categorization(analyzer):
    """Test transaction categorization"""
    categories = analyzer.df['transaction_category'].unique()
    assert 'PAYMENT' in categories
    assert 'INCOME' in categories
    assert 'SUBSCRIPTION' in categories

def test_statistics_calculation(analyzer):
    """Test statistical calculations"""
    assert 'total_transactions' in analyzer.statistics
    assert analyzer.statistics['total_transactions'] == 4
    assert 'total_credits' in analyzer.statistics
    assert analyzer.statistics['total_credits'] == 5000  # Salary amount

def test_context_preparation(analyzer):
    """Test context preparation for LLM"""
    context = analyzer.get_context_for_llm("Show me recent transactions")
    assert 'statistics' in context
    assert 'relevant_transactions' in context
    assert len(context['relevant_transactions']) <= 10  # Should respect limit

@pytest.mark.asyncio
async def test_csv_qa_tool():
    """Test the complete CSV Q&A tool"""
    questions = [
        "What are my recent transactions?",
        "Show me my highest value transactions",
        "What are my spending patterns?",
        "Show me my subscription payments"
    ]
    
    for question in questions:
        response = await csv_qa_llm_tool(question)
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Error" not in response

if __name__ == '__main__':
    pytest.main([__file__])
