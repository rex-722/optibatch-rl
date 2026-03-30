import sys
import os
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import app

#main() function 
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)
  
if __name__ == "__main__":
    main()
