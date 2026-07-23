.PHONY: setup test benchmark ablation compression privacy scaling figures all clean

PY ?= python3
SEEDS ?= 42 43 44 45 46 47 48 49 50 51
DATASET ?= diabetes
DEVICE ?= auto

setup:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -q

benchmark:
	$(PY) -m src.experiments.run_benchmark --dataset $(DATASET) --seeds $(SEEDS) --device $(DEVICE)

ablation:
	$(PY) -m src.experiments.run_ablation --kind all --dataset $(DATASET) --seeds $(SEEDS) --device $(DEVICE)

compression:
	$(PY) -m src.experiments.run_compression --dataset $(DATASET) --device $(DEVICE)

privacy:
	$(PY) -m src.experiments.run_privacy --dataset $(DATASET) --device $(DEVICE)

scaling:
	$(PY) -m src.experiments.run_scaling --dataset $(DATASET) --device $(DEVICE)

figures:
	$(PY) -m src.analysis.make_figures

all: benchmark ablation compression privacy scaling figures

clean:
	rm -rf results/logs/* results/tables/* results/figures/* __pycache__ .pytest_cache
