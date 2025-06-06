import asyncio
from typing import Optional, List
from financial_advisor.sub_agents.data_analyst.enhanced_csv_tool import csv_qa_llm

class FinancialAnalysisSession:
    """Manages an interactive financial analysis session"""
    
    EXAMPLE_QUESTIONS: List[str] = [
        "What are my top 5 largest transactions in the last month?",
        "Show me my spending patterns for UPI payments vs bill payments",
        "Analyze my monthly spending trend and highlight any unusual patterns",
        "What are my most frequent transaction types and their total values?",
        "Compare my recent spending with my historical average"
    ]
    
    def __init__(self):
        self.questions_asked = []
    
    def get_context_aware_response(self, question: str) -> str:
        """Consider previous questions for better context"""
        context = ""
        if self.questions_asked:
            context = f"Previously asked about: {', '.join(self.questions_asked[-3:])}\n"
        return f"{context}New question: {question}"
    
    async def analyze_question(self, question: str) -> str:
        """Process a question through the ADK tool"""
        try:
            self.questions_asked.append(question)
            response = await csv_qa_llm.run_async(
                args={"question": self.get_context_aware_response(question)},
                tool_context=None
            )
            return response
        except Exception as e:
            return str(e)

def print_welcome():
    """Print welcome message and example questions"""
    print("\n🤖 Advanced Financial Analysis Assistant")
    print("=" * 50)
    print("\nI can help you analyze your transaction data in detail.")
    print("Here are some examples of what you can ask:\n")
    for i, q in enumerate(FinancialAnalysisSession.EXAMPLE_QUESTIONS, 1):
        print(f"{i}. {q}")
    print("\nType 'exit' to quit, 'examples' to see sample questions again.\n")

async def main():
    session = FinancialAnalysisSession()
    print_welcome()
    
    while True:
        try:
            question = input("\n💭 Your question: ").strip()
            
            if question.lower() in ('exit', 'quit'):
                print("\nThank you for using the Financial Analysis Assistant! 👋")
                break
            elif question.lower() == 'examples':
                print_welcome()
                continue
            elif not question:
                continue
            
            print("\n🔍 Analyzing your financial data...")
            response = await session.analyze_question(question)
            print(f"\n{response}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try a different question.\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
