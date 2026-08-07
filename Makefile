.PHONY: test native android handoff verify status next requirements compile check

test:
	python -m pytest -q

compile:
	python -m compileall -q host scripts tools tests/python

native:
	./scripts/build_host.sh

android:
	./scripts/build_android.sh

handoff:
	python scripts/handoff_check.py --quick

verify:
	python scripts/handoff_check.py

requirements:
	python scripts/render_requirements.py --check

check: compile test requirements handoff

status:
	python scripts/taskctl.py status

next:
	python scripts/taskctl.py next
