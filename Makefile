
dev:
	ls api/app/*.py | entr -n -r fastapi dev api/app/main.py

play:
	ls main.py | entr -n -r python main.py

run-frontent:
	cd frontend && npm run start

run-backend:
	cd api && uvicorn app.main:app --reload