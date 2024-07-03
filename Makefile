
dev:
	ls api/app/*.py | entr -n -r fastapi dev api/app/main.py

play:
	ls main.py | entr -n -r python main.py