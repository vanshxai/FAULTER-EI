import uvicorn
from server.main import app

if __name__ == "__main__":
    print("=" * 50)
    print("NASSENGER 8 DIGITAL TWIN")
    print("=" * 50)
    print("Dashboard: http://localhost:8000")
    print("WebSocket: ws://localhost:8000/ws")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
