import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_fake_kb(company_name= "Glorious Healthcare", industry="Healthcare Provider"):
    prompt = f"""
Create a synthetic knowledge base text for a fictional company named '{company_name}' in the {industry} industry.

Include the following sections:
1. [Company Overview]
2. [Services]
3. [Key Features]
4. [FAQs] (at least  20 Q&A pairs)
5. [Support Information] (e.g., contact hours, email, live chat)

Write it in clear, structured markdown-like text, like a customer support or internal wiki document.
"""
    response = openai.chat.completions.create(
        model = "gpt-4",
        messages = [
            {"role": "system", "content": "You are a helpful assistant that writes technical documentation"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    kb_text = response.choices[0].message.content # ['choices'][0]['message']['content']
    return kb_text 


if __name__ == "__main__":
    kb_output = generate_fake_kb()

    with open("fake_kb.txt", "w") as f:
        f.write(kb_output)

    print("Fake knowledge base saved to 'fake_kb.txt'")
