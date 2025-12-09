from api.rag_engine import MyanmarAgriAssistEngine

engine = MyanmarAgriAssistEngine()

# Test classification
cls = engine.classify_text("မြေဆီဓာတ် ဘယ်လိုသုံးမလဲ?")
print("=== Classification ===")
print(cls)

# Test RAG retrieval + final answer
ans = engine.ask_question("ကောင်းမွန်တဲ့ မြေပြင်ဘယ်လိုပြုလုပ်မလဲ?", top_k=3)
print("\n=== RAG Answer ===")
print(ans["answer"])

print("\n=== Retrieved Contexts ===")
for i, ctx in enumerate(ans["contexts"], start=1):
    print(f"\n--- Context {i} ---")
    print("Category:", ctx["category"])
    print("Instruction:", ctx["instruction"])
    print("Output:", ctx["output"])
    print("Score:", ctx["score"])

