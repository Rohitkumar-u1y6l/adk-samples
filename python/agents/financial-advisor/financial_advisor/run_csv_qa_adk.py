from financial_advisor.sub_agents.data_analyst.agent import data_analyst_agent
import asyncio
from typing import Optional
from google.adk import InvocationContext
from google.adk.tools.tool_context import ToolContext

async def ask_question(question: str) -> str:
    """Run a question through the CSV Q&A agent asynchronously."""
    try:
        invocation_context = InvocationContext()
        tool_context = ToolContext(invocation_context=invocation_context)
        response = await data_analyst_agent.tools[1].run_async(
            args={"question": question}, 
            tool_context=tool_context
        )
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("🤖 Advanced Financial Data Analysis Assistant")
    print("Ask questions about your transaction data (type 'exit' to quit)\n")
    print("Example complex queries:")
    print("1. What's my monthly spending trend over the last quarter and how does it compare to my average?")
    print("2. Show me high-value transactions (above ₹50,000) and any patterns in their occurrence")
    print("3. Which months had unusual spending patterns compared to the typical range?")
    print("4. Can you identify my recurring transactions and their frequency?")
    print("5. What's my spending distribution across different transaction types?\n")
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            if question.lower() in ('exit', 'quit'):
                print("\nThank you for using the Financial Analysis Assistant!")
                break
            elif not question:
                continue
                
            print("\n🔄 Analyzing your financial data...")
            response = asyncio.run(ask_question(question))
            print(f"\n{response}\n")
            
        except KeyboardInterrupt:
            print("\nThank you for using the Financial Analysis Assistant!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again with a different question.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
