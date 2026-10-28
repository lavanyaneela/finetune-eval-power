# make returns 2 for any failed recipe, and 2 is also this tool's "refused" code.
# Targets that run the tool therefore print its code and return 0, so a reader can
# never confuse "the recipe broke" with "the tool declined to answer". Branch on the
# console script, not on make.

PYTHON ?= python3
REPORTS := reports

.PHONY: help install install-all lint test test-all receipts verify experiments report sweep \
        evidence diagrams screenshots recording demo set-handle clean

help:
	@echo "install      editable install with the dev extra"
	@echo "install-all  also install the training extra (only needed to re-capture)"
	@echo "lint         ruff check and format check"
	@echo "test         pytest with junit and coverage reports, skipping the slow tests"
	@echo "test-all     every test, including the ones that need the training extra"
	@echo "receipts     run the tests, re-measure every published figure, check the documents"
	@echo "verify       lint + test + receipts, offline, from a fresh clone"
	@echo "experiments  run the five experiments and print their tables"
	@echo "report       write the HTML report for both arms"
	@echo "sweep        re-capture the committed measurement (needs the training extra)"
	@echo "evidence     re-render the diagram, the screenshots and the terminal replay"
	@echo "set-handle   fill the GITHUB_USERNAME placeholder: make set-handle HANDLE=you"

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-all:
	$(PYTHON) -m pip install -e ".[dev,train]"

lint:
	ruff check src tests tools experiments
	ruff format --check src tests tools experiments

# -m "not slow" deselects the tests that need the training extra, so the collected
# count and the coverage are the same whether or not the extra is installed. Without
# it the published test count changes depending on what happens to be on the machine.
test:
	mkdir -p $(REPORTS)
	$(PYTHON) -m pytest -q -m "not slow" \
		--junitxml $(REPORTS)/junit.xml \
		--cov=finetune_eval_power \
		--cov-report json:$(REPORTS)/coverage.json \
		--cov-report term:skip-covered

test-all:
	$(PYTHON) -m pytest -q

# Depends on `test` rather than trusting whatever is in reports/. The test count
# and the coverage are the only published figures that do not come from the
# committed measurement, and an audit published 999 tests by editing junit.xml and
# running this target. The mtime guard inside collect_metrics cannot catch that,
# because editing the file refreshes its mtime. Running the tests can.
receipts: test
	$(PYTHON) tools/collect_metrics.py --skip-tests
	$(PYTHON) tools/check_numbers.py
	$(PYTHON) tools/check_durations.py

verify: lint receipts
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

diagrams:
	$(PYTHON) tools/render_diagrams.py

screenshots:
	$(PYTHON) tools/capture_screenshots.py

recording:
	$(PYTHON) tools/record_demo.py

evidence: diagrams screenshots recording

demo:
	$(PYTHON) -m finetune_eval_power.cli power --arm vision --claim 2; \
		echo "tool exit code $$?"
	$(PYTHON) -m finetune_eval_power.cli blind --arm vision; \
		echo "tool exit code $$?"

set-handle:
	@test -n "$(HANDLE)" || (echo "usage: make set-handle HANDLE=your-github-handle" && exit 1)
	$(PYTHON) tools/set_handle.py "$(HANDLE)"

clean:
	rm -rf $(REPORTS) .pytest_cache .ruff_cache .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
