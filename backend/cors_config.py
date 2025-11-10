# cors_config.py
from fastapi.middleware.cors import CORSMiddleware

# Define which origins are allowed to connect.
# In a production environment, you would restrict this to your actual frontend URL.
origins = [
    "http://localhost",
    "http://localhost:3000",    # Default for create-react-app
    "http://localhost:5173",    # Default for Vite
    "http://127.0.0.1:5173",
    "https://buddy-j3f5.vercel.app",  # <-- Add your Vercel frontend URL here
    "https://test-f-ochre.vercel.app/",
    "https://ecom-chi-lac.vercel.app",
    "https://saathi-ai.vercel.app"
]

def setup_cors(app):
    """
    Configures and adds CORS middleware to the FastAPI application.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_credentials=True,
        allow_methods=["*"],    # Allows all methods (GET, POST, etc.)
        allow_headers=["*"],    # Allows all headers
    )

