# brief_local.py
# -*- coding: utf-8 -*-

import json
import time
import subprocess
from pathlib import Path
import os
import psutil
import re
import platform
import socket

SYSTEM_PROMPT = """\
너는 한국어로만 응답하는 금융 애널리스트다.
절대로 중국어, 영어를 사용하지 말고 반드시 한국어로만 작성하라.

다음은 '신한투자증권 및 신한금융그룹 관련 최신 뉴스 기사 전문'이다.

이 기사들을 종합해
"투자자용 데일리 기업 브리핑"을 작성하라.
너는 질문은 하지마.

[출력 형식]
1. 한 줄 결론 (한국어)
2. 핵심 요약 5줄 (한국어, 중복 제거)

[중요 규칙]
- 출력 언어: 한국어 ONLY
- 기사에 없는 사실을 추정하지 말 것
- 간결하지만 분석적으로
"""

# ====== CONFIG ======
INPUT_JSONL = Path("out_shinhan/articles.jsonl")
OUTPUT_DIR = Path("out_shinhan")
OUTPUT_SUMMARY_JSON = OUTPUT_DIR / "summary.json"
OUTPUT_PERF_JSON = OUTPUT_DIR / "perf.json"

MODEL_NAME = "mistral:7b-instruct-q4_0"
MAX_ARTICLES = 10

# Ollama가 구형이면 run 옵션이 제한될 수 있어서,
# stdin으로 prompt 전달하는 방식(현재 너가 쓰는 방식)이 제일 호환성 좋음.
# ====================


def load_articles(jsonl_path: Path, max_articles=10):
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input jsonl not found: {jsonl_path}")

    articles = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            articles.append(json.loads(line))
            if len(articles) >= max_articles:
                break
    return articles


def build_prompt(articles):
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(f"[기사 {i}]")
        lines.append(f"제목: {a.get('title','')}")
        lines.append(f"언론사: {a.get('press','')}")
        lines.append(f"발행일: {a.get('published','')}")
        lines.append(f"본문:\n{a.get('text','')}\n")
    return SYSTEM_PROMPT + "\n\n" + "\n".join(lines)


def run_ollama(model: str, prompt: str) -> str:
    # returncode != 0이면 stderr에 에러가 옴 (EOF 등)
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ollama failed with empty stderr")
    return proc.stdout.strip()


def estimate_char_count(s: str) -> int:
    return len(s)


def estimate_korean_ratio(text: str) -> float:
    # 한글 비율 대충 체크(중국어/영어 튀는지 감지용)
    if not text:
        return 0.0
    hangul = len(re.findall(r"[가-힣]", text))
    return hangul / max(1, len(text))


def get_env_info() -> dict:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "arch": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ✅ 프로세스 / 성능 측정 준비
    process = psutil.Process(os.getpid())
    cpu_times_start = process.cpu_times()
    rss_start = process.memory_info().rss

    # ✅ 전체 시간 측정
    t0 = time.perf_counter()

    # ✅ 입력/프롬프트 구성 시간
    t_load0 = time.perf_counter()
    articles = load_articles(INPUT_JSONL, MAX_ARTICLES)
    prompt = build_prompt(articles)
    t_load1 = time.perf_counter()

    # ✅ LLM 호출 시간(핵심 측정 구간)
    t_llm0 = time.perf_counter()
    summary_text = run_ollama(MODEL_NAME, prompt)
    t_llm1 = time.perf_counter()

    # ✅ 종료 측정
    t1 = time.perf_counter()
    elapsed_total = t1 - t0
    elapsed_load = t_load1 - t_load0
    elapsed_llm = t_llm1 - t_llm0

    # ✅ CPU 시간 / 메모리 측정
    cpu_times_end = process.cpu_times()
    cpu_user = cpu_times_end.user - cpu_times_start.user
    cpu_system = cpu_times_end.system - cpu_times_start.system
    cpu_total = cpu_user + cpu_system

    rss_end = process.memory_info().rss
    rss_peak_like = max(rss_start, rss_end)  # 단순 비교(진짜 peak는 아님)
    mem_mb_end = rss_end / (1024 ** 2)

    # CPU 사용률 "추정치"
    # (단일 프로세스가 전체 시간 동안 쓴 CPU 시간 / wall time) * 100
    cpu_usage_est = (cpu_total / elapsed_total) * 100 if elapsed_total > 0 else 0.0

    # ✅ 처리량 지표(LLM 측정에 유용)
    prompt_chars = estimate_char_count(prompt)
    out_chars = estimate_char_count(summary_text)

    prompt_chars_per_s = prompt_chars / elapsed_llm if elapsed_llm > 0 else 0.0
    out_chars_per_s = out_chars / elapsed_llm if elapsed_llm > 0 else 0.0

    # ✅ 결과 저장(요약)
    summary_payload = {
        "model": MODEL_NAME,
        "articles_used": len(articles),
        "elapsed_seconds_total": round(elapsed_total, 3),
        "elapsed_seconds_load": round(elapsed_load, 3),
        "elapsed_seconds_llm": round(elapsed_llm, 3),
        "summary": summary_text,
    }
    OUTPUT_SUMMARY_JSON.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ✅ 결과 저장(성능)
    perf_payload = {
        "env": get_env_info(),
        "input": {
            "jsonl": str(INPUT_JSONL),
            "max_articles": MAX_ARTICLES,
        },
        "model": MODEL_NAME,
        "timing": {
            "total_s": round(elapsed_total, 3),
            "load_s": round(elapsed_load, 3),
            "llm_s": round(elapsed_llm, 3),
        },
        "cpu": {
            "user_s": round(cpu_user, 3),
            "system_s": round(cpu_system, 3),
            "total_s": round(cpu_total, 3),
            "usage_est_percent": round(cpu_usage_est, 2),
        },
        "memory": {
            "rss_end_mb": round(mem_mb_end, 2),
            "rss_start_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "rss_peak_like_bytes": rss_peak_like,
        },
        "throughput": {
            "prompt_chars": prompt_chars,
            "output_chars": out_chars,
            "prompt_chars_per_s": round(prompt_chars_per_s, 2),
            "output_chars_per_s": round(out_chars_per_s, 2),
        },

    }
    OUTPUT_PERF_JSON.write_text(
        json.dumps(perf_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ✅ 콘솔 출력(README에 그대로 붙이기 좋게)
    print("✅ DONE")
    print(f"- model: {MODEL_NAME}")
    print(f"- articles_used: {len(articles)}")
    print(f"- total: {elapsed_total:.3f}s (load {elapsed_load:.3f}s + llm {elapsed_llm:.3f}s)")
    print(f"- cpu_total: {cpu_total:.3f}s (est usage {cpu_usage_est:.2f}%)")
    print(f"- rss_end: {mem_mb_end:.2f}MB")
    print(f"- throughput: prompt {prompt_chars_per_s:.2f} chars/s, output {out_chars_per_s:.2f} chars/s")
    print(f"- saved: {OUTPUT_SUMMARY_JSON} , {OUTPUT_PERF_JSON}")


if __name__ == "__main__":
    main()