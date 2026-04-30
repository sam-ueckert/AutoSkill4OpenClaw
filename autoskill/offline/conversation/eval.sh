export DASHSCOPE_API_KEY="sk-8480a5c7f4e041678f109615de9ea0de"

python -m autoskill.offline.conversation.self_evolve.eval \
  --run-root log/self-evolve/2026-0416-1233/Reflection \
  --prompt-source best \
  --prompt-file log/self-evolve/2026-0416-1233/Reflection/manual_prompt.txt \
  --dataset eval \
  --eval-root data/eval \
  --eval-meta-info-jsonl data/eval/meta_info.jsonl \
  --eval-name eval_manual \
  --extract-mode specific \
  --max-workers 16 \
  --max-failed-retries 5 \
  --disable-env-proxy 1 \
  --codex-auto-backend llm \
  --llm-provider codex \
  --llm-model gpt-5.2 \
  --embeddings-provider dashscope
