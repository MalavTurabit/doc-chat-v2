from config import (
    AZURE_OPENAI_EMB_KEY, AZURE_EMB_ENDPOINT, AZURE_EMB_API_VERSION, AZURE_EMB_DEPLOYMENT,
    AZURE_OPENAI_LLM_KEY, AZURE_LLM_ENDPOINT, AZURE_LLM_API_VERSION, AZURE_LLM_DEPLOYMENT,
    EMBEDDING_DIM,
)

checks = {
    "EMB_KEY":        AZURE_OPENAI_EMB_KEY,
    "EMB_ENDPOINT":   AZURE_EMB_ENDPOINT,
    "EMB_VERSION":    AZURE_EMB_API_VERSION,
    "EMB_DEPLOYMENT": AZURE_EMB_DEPLOYMENT,
    "LLM_KEY":        AZURE_OPENAI_LLM_KEY,
    "LLM_ENDPOINT":   AZURE_LLM_ENDPOINT,
    "LLM_VERSION":    AZURE_LLM_API_VERSION,
    "LLM_DEPLOYMENT": AZURE_LLM_DEPLOYMENT,
}

all_ok = True
for name, val in checks.items():
    status = "OK" if val else "MISSING"
    if not val:
        all_ok = False
    print(f"  {status}  {name}: {val[:20]}..." if val else f"  {status}  {name}")

print(f"\nEmbedding dim : {EMBEDDING_DIM}")
print(f"\n{'All config loaded.' if all_ok else 'Fix missing values above.'}")