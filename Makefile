.PHONY: install test bench-cleanup bench-e2e live serve doctor

install:
	pip install -e ".[all]"

test:
	pytest -q

doctor:
	flowlite doctor

# Text-only benchmark: raw -> rules -> LLM (needs Ollama or any OpenAI-compatible server for the llm row)
bench-cleanup:
	flowlite bench-cleanup --backends raw,rules,llm --repeats 3

# Audio benchmark: needs data/audio/<id>.wav (see scripts/record_dataset.py)
bench-e2e:
	flowlite bench-e2e --backends raw,rules,llm

live:
	flowlite live

serve:
	flowlite serve
