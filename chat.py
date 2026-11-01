"""
chat.py - talk to the handbook in your terminal.

    python chat.py
Commands:  quit  |  sources (toggle showing retrieved chunks)
"""
from rag import HandbookRAG

print("Loading model and database ...")
rag = HandbookRAG()
print("Ready! Ask anything about the handbook. Type 'quit' to exit, "
      "'sources' to toggle sources.\n")

history = []
show_sources = True

while True:
    question = input("You: ").strip()
    if not question:
        continue
    if question.lower() in ("quit", "exit"):
        break
    if question.lower() == "sources":
        show_sources = not show_sources
        print(f"(sources {'on' if show_sources else 'off'})\n")
        continue

    result = rag.answer(question, history)
    print(f"\nBot: {result['answer']}\n")

    if show_sources:
        print("  Retrieved chunks:")
        for h in result["sources"]:
            print(f"   {h['score']:.3f} | p.{h['page']:<3} | {h['text'][:90]}...")
        print()

    history += [{"role": "user", "content": question},
                {"role": "assistant", "content": result["answer"]}]
