.PHONY: fmt
fmt:
	isort . \
	&& black .

.PHONY: lint
lint:
	pre-commit run --all
	# black --check .
	# isort --check .
	# flake8 .
	# mypy .

.PHONY: test
test:
	pytest --verbose

.PHONY: install
install:
	pip install .

.PHONY: install-dev
install-dev:
	pip install -r requirements.txt \
	&& pip install -e .

.PHONY: clean-docs
clean-docs:
	rm -rf docs

.PHONY: docs
docs: clean-docs
	mkdir -p docs \
	&& pdoc --html --output-dir docs fibsemtools

.PHONY: readme
readme:
	dathead --help | p2c --tgt _dathead README.md \
	&& datview --help | p2c --tgt _datview README.md \
	&& dathist --help | p2c --tgt _dathist README.md \
	&& datcalib --help | p2c --tgt _datcalib README.md
