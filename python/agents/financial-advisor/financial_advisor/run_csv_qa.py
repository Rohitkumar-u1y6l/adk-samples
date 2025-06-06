import sys
from financial_advisor.sub_agents.data_analyst import answer_question_about_csv

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_csv_qa.py 'your question about the CSV file'")
        sys.exit(1)
    question = sys.argv[1]
    answer = answer_question_about_csv(question)
    print(answer)

if __name__ == "__main__":
    main()
