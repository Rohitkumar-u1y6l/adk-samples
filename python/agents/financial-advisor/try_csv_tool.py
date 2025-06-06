import asyncio
import os
from pathlib import Path
from financial_advisor.sub_agents.data_analyst.csv_tool import csv_qa_llm_tool

async def main():
    # Verify data file exists
    data_path = Path('/workspaces/adk-samples/data/transactions.csv')
    print(f"Checking for data file at: {data_path}")
    if not data_path.exists():
        print(f"❌ Error: Could not find transaction data at {data_path}")
        return
    else:
        print(f"✓ Found transaction data file")

    print("\n🔍 Enhanced CSV Transaction Analyzer")
    print("------------------------------------------")
    print("Type 'quit' to exit")
    print("\nExample questions you can ask:")
    print("- What are my top 5 highest value transactions?")
    print("- Show me my recent UPI payments")
    print("- What are my monthly spending patterns?")
    print("- Show me my subscription payments")
    print("- What's my spending trend over weekends vs weekdays?")
    
    while True:
        try:
            print("\n💭 Your question: ", end='')
            question = input().strip()
            
            if question.lower() in ('quit', 'exit', 'q'):
                print("\n✨ Thanks for using the Transaction Analyzer!")
                break
                
            if not question:
                continue
                
            print("\n🔍 Analyzing...")
            try:
                print("Calling csv_qa_llm_tool...")
                response = await csv_qa_llm_tool(question)
                print(f"\n{response}")
            except Exception as e:
                print(f"\n❌ Analysis error: {str(e)}")
                import traceback
                print("Traceback:")
                traceback.print_exc()
            
        except KeyboardInterrupt:
            print("\n\n✨ Thanks for using the Transaction Analyzer!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            print("Traceback:")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
    finally:
        print("\n✨ Session ended")
