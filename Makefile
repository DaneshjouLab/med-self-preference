# med-self-preference - common commands
# Run from project root: make <target>

.PHONY: test help

help:
	@echo "med-self-preference targets:"
	@echo "  make test          - Run smoke test (verify API keys, dataset load)"
	@echo "  make covid-parse   - Parse COVID dataset only (20 scenarios)"
	@echo "  make covid-gen     - Generate single-turn COVID responses (100 scenarios, gpt-4o)"
	@echo "  make pairwise-single - Show pairwise_evaluation_single.py help"

test:
	python src/test_generation.py

covid-parse:
	python src/generation/generate_single_turn_covid.py \
		--source_file ./COVID-Dialogue-Dataset-English.txt \
		--num_scenarios 20 \
		--parse_only

covid-gen:
	python src/generation/generate_single_turn_covid.py \
		--source_file ./COVID-Dialogue-Dataset-English.txt \
		--num_scenarios 100 \
		--models gpt-4o \
		--output_dir ./covid_dialogue_output

pairwise-single:
	python src/evaluation/pairwise_evaluation_single.py --help
