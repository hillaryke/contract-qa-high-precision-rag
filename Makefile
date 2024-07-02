
dev:
	ls api/app/*.py | entr -n -r fastapi dev api/app/main.py