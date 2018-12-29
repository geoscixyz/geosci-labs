.PHONY: tests clean

tests:
	nosetests --logging-level=INFO

clean:
	find . -name "*.pyc" | xargs -I {} rm -v "{}"
