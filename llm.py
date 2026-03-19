import ollama

def generate_response(prompt):
    response =ollama.chat(
        model="phi3:mini",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        stream=True
    )
    result=""
    for chunk in response:
        token=chunk["message"]["content"]
        print(token, end="", flush=True)
        result+=token
    return result

if __name__ == "__main__":
    print(generate_response("hello"))