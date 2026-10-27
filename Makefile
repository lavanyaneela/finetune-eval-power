# make returns 2 for any failed recipe, and 2 is also this tool's "refused" code.
# Targets that run the tool therefore print its code and return 0, so a reader can
# never confuse "the recipe broke" with "the tool declined to answer". Branch on the
# console script, not on make.

PYTHON ?= python3
REPORTS := reports

.PHONY: help install lint test receipts verify experiments report sweep demo clean

help:
	@echo "install      editable install with the dev extra"
	@echo "install-all  also install the training extra (only needed to re-capture)"
	@echo "lint         ruff check and format check"
	@echo "test         pytest with junit and coverage reports"
	@echo "receipts     re-measure every published figure and check the documents"
	@echo "verify       lint + test + receipts, offline, from a fresh clone"
	@echo "experiments  run the five experiments and print their tables"
	@echo "report       write the HTML report for both arms"
	@echo "sweep        re-capture the committed measurement (needs the training extra)"

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-all:
	$(PYTHON) -m pip install -e ".[dev,train]"

lint:
	ruff check src tests tools experiments
	ruff format --check src tests tools experiments

test:
	mkdir -p $(REPORTS)
	$(PYTHON) -m pytest -q \
		--junitxml $(REPORTS)/junit.xml \
		--cov=finetune_eval_power \
		--cov-report json:$(REPORTS)/coverage.json \
		--cov-report term:skip-covered

receipts:
	$(PYTHON) tools/collect_metrics.py --skip-tests
	$(PYTHON) tools/check_numbers.py
	$(PYTHON) tools/check_durations.py

verify: lint test receipts
	@echo "verify passed"

experiments:
	@for name in e1_detection_floor e2_forgetting_curve e3_blind_capabilities \
	             e4_sequence_arm e5_freeze_depth; do \
		$(PYTHON) experiments/$$name.py; \
	done

report:
	$(PYTHON) -m finetune_eval_power.cli report --arm vision --out $(REPORTS)/vision.html; \
		echo "exit code $$?"
	$(PYTHON) -m finetune_eval_power.cli report --arm sequence --out $(REPORTS)/sequence.html; \
		echo "exit code $$?"

sweep:
	$(PYTHON) tools/capture_sweep.py

demo:
	$(PYTHON) -m finetune_eval_power.cli power --arm vision --claim 2; \
		echo "tool exit code $$?"
	$(PYTHON) -m finetune_eval_power.cli blind --arm vision; \
		echo "tool exit code $$?"

clean:
	rm -rf $(REPORTS) .pytest_cache .ruff_cache .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
