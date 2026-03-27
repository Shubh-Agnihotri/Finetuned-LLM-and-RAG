from pipeline import ask_rag

print("Ask something:")
query = input()

while query != "exit":
    response = ask_rag(query)
    print("Response:", response)
    print("\nType 'exit' to stop or continue asking:")
    query = input()
