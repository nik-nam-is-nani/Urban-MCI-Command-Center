import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import app

def main():
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)

if __name__ == "__main__":
    main()
