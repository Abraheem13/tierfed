.PHONY: setup test determinism benchmark ablation compression privacy scaling collect figures paper all clean

PY ?= python
DATASET ?= diabetes
DEVICE ?= cuda

setup:
	bash scripts/setup_env.sh venv

test:
	$(PY) -m pytest tests -q

determinism:
	bash scripts/check_determinism.sh $(DATASET) cpu

benchmark:
	STAGES="2" bash scripts/reproduce_all.sh
ablation:
	STAGES="3 4" bash scripts/reproduce_all.sh
privacy:
	STAGES="5" bash scripts/reproduce_all.sh
scaling:
	STAGES="6" bash scripts/reproduce_all.sh
collect:
	$(PY) -m src.analysis.collect --results results --dataset $(DATASET) --model mlp
figures:
	$(PY) -m src.analysis.make_figures --results results --dataset $(DATASET) --model mlp
paper: collect figures
	$(PY) scripts/claim_check.py paper/generated/flags.json paper/claims.json
	$(MAKE) -C paper

all:
	bash scripts/reproduce_all.sh

clean:
	rm -rf results/logs/* results/tables/* results/determinism __pycache__ .pytest_cache
