# cors_config.py
from fastapi.middleware.cors import CORSMiddleware

# List of allowed frontend URLs (production and development)
origins = [
    "http://localhost",
    "http://localhost:3000",    # Create-React-App dev
    "http://localhost:5173",    # Vite dev
    "http://127.0.0.1:5173",
    "https://buddy-j3f5.vercel.app",
    "https://test-f-ochre.vercel.app",
    "https://ecom-chi-lac.vercel.app",
    "https://saathi-ai.vercel.app"
]

def setup_cors(app):
    """
    Configures and adds CORS middleware to the FastAPI application.
    Must explicitly list origins if allow_credentials=True.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,   # <-- Explicit list is required when using credentials
        allow_credentials=True,  # Required if frontend sends cookies/auth headers
        allow_methods=["*"],     # Allow all HTTP methods
        allow_headers=["*"],     # Allow all headers
    )
