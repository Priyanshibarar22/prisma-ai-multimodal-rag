import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
from search import search

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def answer_query(query, k=8):
    if not os.environ.get("GROQ_API_KEY"):
        return "GROQ_API_KEY is not set. Please set it before asking questions.", []

    retrieved = search(query, k=k)
    if not retrieved:
        return "No relevant content found. Have you processed any files yet?", []

    context = "\n\n".join(
        f"[{r['title']}, {r['start']}s-{r['end']}s]: {r['text']}"
        for r in retrieved
    )

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have that information in the uploaded content."

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content, retrieved
    except Exception as e:
        return f"Could not get a response from the LLM: {e}", retrieved

