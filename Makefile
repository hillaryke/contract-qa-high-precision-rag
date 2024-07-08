
dev:
	ls api/app/*.py | entr -n -r fastapi dev api/app/main.py

play:
	ls main.py | entr -n -r python main.py

run-frontend:
	cd frontend && npm run dev

run-backend:
	cd backend/app && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install