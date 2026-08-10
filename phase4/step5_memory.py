from retail_agent import agent, build_deps
from step3_output import AnalysisResult


deps = build_deps()

# Tur 1
result1 = agent.run_sync(
    "42 numaralı mağazada fiyat anormalliği var mı?",
    deps=deps,
    output_type=AnalysisResult,
)
print("=== TUR 1 ===")
print(result1.output)


# Tur 2 — message_history yok
result2 = agent.run_sync(
    "peki bu işlemlerin toplam tutarı ne kadar?",
    deps=deps,
    output_type=AnalysisResult,
)
print("\n=== TUR 2 ===")
print(result2.output)


# Tur 3 — Tur 1'in geçmişiyle
result3 = agent.run_sync(
    "peki bu işlemlerin toplam tutarı ne kadar?",
    deps=deps,
    output_type=AnalysisResult,
    message_history=result1.all_messages(),
)
print("\n=== TUR 3 ===")
print(result3.output)

print("\n=== MESSAGE COUNTS ===")
print("len(all_messages()):", len(result3.all_messages()))
print("len(new_messages()):", len(result3.new_messages()))