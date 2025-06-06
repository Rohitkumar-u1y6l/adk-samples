# Utility functions for reading and analyzing CSV files for the financial advisor agent
import pandas as pd
from typing import Optional
import re
import datetime

def load_transactions_csv(csv_path: str = "/workspaces/adk-samples/data/transactions.csv") -> pd.DataFrame:
    """Load the transactions CSV file into a pandas DataFrame."""
    return pd.read_csv(csv_path)


def answer_csv_question(df: pd.DataFrame, question: str) -> str:
    """
    Answer a question about the transactions DataFrame using pandas.
    This is a placeholder for more advanced logic or LLM integration.
    """
    # Try to extract money values from both 'moneyValue' and text columns
    money = pd.to_numeric(df["moneyValue"], errors="coerce")
    # Try to extract money values from 'mentionText' column if present
    if "mentionText" in df.columns:
        def extract_money(text):
            if pd.isna(text):
                return 0.0
            matches = re.findall(r"[\d,]+\.\d{2}", str(text))
            return sum(float(m.replace(",", "")) for m in matches)
        mention_money = df["mentionText"].apply(extract_money)
        money = money.fillna(0) + mention_money.fillna(0)
    total = money.sum()
    # Normalize question for flexible matching
    q = question.lower().replace("spent", "spend").replace("so far", "").replace("so dar", "")
    if ("total" in q or "sum" in q or "spend" in q or "amount" in q or "inr" in q):
        return (
            "Based on your transaction records, the total money value is: "
            f"**₹{total:,.2f}** INR.\n\n"
            "This includes all amounts found in both structured and unstructured fields of your CSV. "
            "If you need a breakdown or more details, just ask!"
        )
    if "count" in q:
        return (
            f"You have **{len(df):,}** transactions recorded in your CSV.\n\n"
            "Let me know if you want to analyze a specific type or time period!"
        )
    # Yearly breakdown for last 3 years
    if ("per year" in q or "yearly" in q or "each year" in q) and ("last 3" in q or "past 3" in q or "three year" in q):
        # Try to parse dateValue or mentionText for year
        if "dateValue" in df.columns:
            # Try to parse year from dateValue
            def extract_year(val):
                try:
                    return datetime.datetime.strptime(str(val), "%d/%m/%y").year
                except Exception:
                    return None
            years = df["dateValue"].apply(extract_year)
        else:
            years = None
        if years is not None:
            df["_year"] = years
            recent_years = sorted([y for y in years.unique() if y is not None and not pd.isna(y)], reverse=True)[:3]
            breakdown = []
            for y in recent_years:
                year_sum = pd.to_numeric(df[df["_year"] == y]["moneyValue"], errors="coerce").sum()
                # Add mentionText extraction
                if "mentionText" in df.columns:
                    mention_money = df[df["_year"] == y]["mentionText"].apply(extract_money).sum()
                    year_sum += mention_money
                breakdown.append(f"{y}: **₹{year_sum:,.2f}** INR")
            return (
                "Here is your spending per year for the last 3 years (from your CSV):\n\n" +
                "\n".join(breakdown) +
                "\n\nLet me know if you want a different breakdown or more details!"
            )
        else:
            return "Sorry, I couldn't extract years from your data. Please ensure 'dateValue' column is present and formatted as DD/MM/YY."
    return (
        "I'm sorry, I couldn't understand your question. "
        "Please ask about totals, counts, or specify a column or type of transaction you want to analyze."
    )
