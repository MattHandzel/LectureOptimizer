import ollama

OLLAMA_NUM_THREADS = 14


def query_ollama(user_message, system_message, model_name="llama3.2"):

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        options={"num_thread": OLLAMA_NUM_THREADS},
    )
    return response["message"]["content"].strip()
