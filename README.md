# Lab 16: Reflexion Agent

**Sinh viên:** Ngô Anh Tuấn<br>
**Môi trường kiểm chứng:** Python 3.14.5

## Tổng quan

Lab triển khai và so sánh hai agent trên bài toán multi-hop QA:

- **ReAct Agent:** một lượt Actor và Evaluator.
- **Reflexion Agent:** khi câu trả lời sai, Reflector chẩn đoán nguyên nhân,
  ghi lesson/strategy vào reflection memory và thử lại.

```text
Question + Context
        |
        v
      Actor
        |
        v
 Structured Evaluator ---- correct ----> Final answer
        |
       wrong
        v
     Reflector
        |
        v
 Reflection Memory -> retry Actor
```

Runtime hỗ trợ hai chế độ:

- `live`: gọi OpenAI thật cho Actor, Evaluator và Reflector.
- `mock`: deterministic, không dùng API, phục vụ test và autograding.

## Kết quả hiện tại

- Toàn bộ **7.405** mẫu HotpotQA đã được chuyển sang `QAExample`.
- **8/8 tests pass**.
- Mock benchmark chạy 100 mẫu cho mỗi agent, tổng cộng 200 records.
- `autograde.py`: **100/100** gồm 80/80 core và 20/20 bonus.

Kết quả mock benchmark:

| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| Exact Match | 0.76 | **1.00** | **+0.24** |
| Avg attempts | 1.00 | 1.24 | +0.24 |
| Avg token accounting | 1440.43 | 1814.23 | +373.80 |
| Avg latency (ms) | 2.00 | 2.72 | +0.72 |

Mock mode cố ý tạo một tập failure ổn định để kiểm tra Reflexion recovery.
Số liệu live được ghi riêng vì token và latency phụ thuộc OpenAI.

## Dữ liệu

File HotpotQA gốc đã được chuyển thành:

```text
data/hotpot_qaexamples.json
```

Mỗi record tuân thủ schema:

```json
{
  "qid": "5a8b57f25542995d1e6f1371",
  "difficulty": "hard",
  "question": "Multi-hop question",
  "gold_answer": "Gold answer",
  "context": [
    {
      "title": "Document title",
      "text": "Joined paragraph sentences"
    }
  ]
}
```

Converter tái sử dụng nằm tại `scripts/convert_hotpot.py`. Nó kiểm tra required
fields, difficulty, context và duplicate `qid`, sau đó ghi output atomic.

## Core Flow

### Structured schemas

`JudgeResult` gồm:

- binary `score`;
- `reason`;
- failure mode;
- missing evidence;
- spurious claims;
- confidence;
- token count và latency.

`ReflectionEntry` gồm root cause, transferable lesson, next strategy, evidence
cần kiểm tra và usage của Reflector.

### Actor

Actor chỉ dùng context, hoàn thành toàn bộ hop và trả JSON gồm reasoning summary
cùng final answer ngắn.

### Evaluator

Evaluator là LLM judge có structured JSON output. Nó phân biệt:

- `entity_drift`;
- `incomplete_multi_hop`;
- `wrong_final_answer`;
- `looping`;
- `reflection_overfit`.

### Reflector

Reflector không được sao chép gold answer. Nó biến evaluator feedback thành
lesson và chiến thuật có thể dùng cho attempt kế tiếp.

### Actual accounting

Trong live mode:

- token lấy từ `response.usage.total_tokens`;
- latency đo bằng `time.perf_counter()` quanh từng API call;
- một attempt cộng Actor + Evaluator + Reflector nếu có.

Không còn token hoặc latency hardcoded trong agent loop.

## Bonus Extensions

Đã triển khai sáu extension được rubric công nhận:

1. `structured_evaluator`: Pydantic validation cho JSON judge.
2. `reflection_memory`: lesson và strategy được đưa lại vào Actor.
3. `adaptive_max_attempts`: easy/medium/hard có attempt budget khác nhau.
4. `memory_compression`: giới hạn và nén reflection cũ.
5. `benchmark_report_json`: xuất report JSON và Markdown.
6. `mock_mode_for_autograding`: benchmark deterministic không cần API.

Rubric chỉ tính tối đa hai extension, tương ứng **20/20 bonus**.

## Cài đặt

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Điền OpenAI key:

```dotenv
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EVALUATOR_MODEL=gpt-4o-mini
```

`.env` và `outputs/` không được commit.

## Chạy tests

```bash
source .venv/bin/activate
python -m pytest -v
```

Tests kiểm tra:

- Unicode answer normalization;
- dataset đúng `QAExample`;
- Judge/Reflection schemas;
- ReAct chỉ dùng một attempt;
- Reflexion ghi memory và recovery;
- adaptive attempts;
- bounded memory compression;
- report đạt toàn bộ điều kiện autograde.

## Chạy benchmark

Benchmark deterministic 100 mẫu:

```bash
python run_benchmark.py \
  --dataset data/hotpot_qaexamples.json \
  --out-dir reports/mock_100 \
  --mode mock \
  --limit 100 \
  --workers 8
```

Benchmark OpenAI thật:

```bash
python run_benchmark.py \
  --dataset data/hotpot_qaexamples.json \
  --out-dir reports/live_100 \
  --mode live \
  --model gpt-4o-mini \
  --limit 100 \
  --workers 6
```

Live benchmark có thể gọi API hàng trăm lần và phát sinh chi phí. ReAct và
Reflexion luôn chạy trên cùng subset theo cùng thứ tự.

Kiểm tra Actor, Evaluator và Reflector bằng bốn API calls trước khi chạy full:

```bash
python scripts/live_smoke.py
```

## Chấm điểm

```bash
python autograde.py --report-path reports/mock_100/report.json
python check_lab.py
```

Điều kiện đã đáp ứng:

| Rubric | Bằng chứng |
|---|---|
| 6 report keys | `report.json` |
| ReAct + Reflexion | `summary` và JSONL runs |
| >=100 records | 200 records |
| >=20 detailed examples | 40 examples, cân bằng hai agent |
| >=3 failure modes | 5-mode taxonomy có diagnosis và fix |
| Discussion >=250 chars | trade-off analysis trong report |
| >=2 extensions | 6 extensions đã triển khai |

## Cấu trúc repo

```text
Lab16/
├── README.md
├── requirements.txt
├── run_benchmark.py
├── autograde.py
├── check_lab.py
├── scripts/
│   └── convert_hotpot.py
├── data/
│   ├── hotpot_mini.json
│   └── hotpot_qaexamples.json
├── src/reflexion_lab/
│   ├── agents.py
│   ├── mock_runtime.py
│   ├── prompts.py
│   ├── reporting.py
│   ├── schemas.py
│   └── utils.py
├── tests/
│   └── test_utils.py
└── reports/
    ├── mock_100/
    └── live_100/
```
