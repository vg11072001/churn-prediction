import pandas as pd
import requests

# Your OpenRouter API credentials
headers = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

def llama3_summarize(prompt: str) -> str:
    """
    Calls LLaMA-3 API via OpenRouter to summarize the given prompt.
    """
    data = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following text:\n\n{prompt}"}
        ]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error calling LLaMA API: {e}")
        return "Failed to generate summary"

def generate_summary(row, pred):
    """
    Constructs a prompt and uses LLaMA-3 API to return a one-liner summary.
    """
    prompt = (
        f"Customer with tenure {row.get('tenure', 0)} months, "
        f"monthly charges ${row.get('MonthlyCharges', 0)}, "
        f"contract type '{row.get('Contract', 'Unknown')}', "
        f"and internet service '{row.get('InternetService', 'Unknown')}' "
        f"is {'likely to churn' if pred else 'likely to stay'}."
    )
    return llama3_summarize(prompt)