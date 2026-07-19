# Project FORESIGHT Automation Commands

.PHONY: data pipeline train risk dashboard api test all

data:
	python src/generate_data.py

pipeline:
	python src/pipeline.py

train:
	python src/forecast.py

risk:
	python src/risk.py

dashboard:
	streamlit run app/dashboard.py

api:
	uvicorn service.main:app --reload

test:
	pytest tests/

all: data pipeline train risk test
