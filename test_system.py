"""
System Test Script

Run this to verify your installation is working correctly.

Usage: python test_system.py
"""

import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("\n" + "="*50)
    print("Testing Package Imports...")
    print("="*50)
    
    packages = {
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "streamlit": "Streamlit",
        "PyPDF2": "PyPDF2",
        "docx": "python-docx",
        "chromadb": "ChromaDB",
        "sentence_transformers": "Sentence Transformers",
        "groq": "Groq",
        "langchain": "LangChain",
        "pydantic": "Pydantic",
        "dotenv": "python-dotenv",
    }
    
    all_good = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            all_good = False
    
    return all_good


def test_env_file():
    """Test if .env file exists and has required keys"""
    print("\n" + "="*50)
    print("Testing Environment Configuration...")
    print("="*50)
    
    if not os.path.exists("backend/.env"):
        print("❌ backend/.env file not found!")
        print("   Create it by copying .env.example:")
        print("   cp .env.example backend/.env")
        return False
    
    print("✅ backend/.env file exists")
    
    # Try to load env
    from dotenv import load_dotenv
    load_dotenv("backend/.env")
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key == "your_groq_api_key_here":
        print("⚠️  GROQ_API_KEY not set or still using placeholder")
        print("   Get your key from: https://console.groq.com/keys")
        print("   Then add it to .env file")
        return False
    
    print("✅ GROQ_API_KEY is set")
    return True


def test_directories():
    """Test if required directories exist"""
    print("\n" + "="*50)
    print("Testing Directory Structure...")
    print("="*50)
    
    dirs = [
        "backend",
        "frontend", 
        "backend/data",
        "backend/data/uploads",
        "backend/data/chromadb"
    ]
    
    all_good = True
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - NOT FOUND")
            all_good = False
    
    return all_good


def test_embedding_model():
    """Test if embedding model can be loaded"""
    print("\n" + "="*50)
    print("Testing Embedding Model...")
    print("="*50)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("🔄 Loading model (first time may take a minute)...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        print("✅ Model loaded successfully")
        
        # Test encoding
        test_text = "This is a test"
        embedding = model.encode([test_text])
        print(f"✅ Test encoding successful (dimension: {len(embedding[0])})")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


def test_groq_connection():
    """Test Groq API connection"""
    print("\n" + "="*50)
    print("Testing Groq API Connection...")
    print("="*50)
    
    try:
        from groq import Groq
        from dotenv import load_dotenv
        
        load_dotenv("backend/.env")
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key or api_key == "your_groq_api_key_here":
            print("⚠️  Cannot test - API key not configured")
            return False
        
        client = Groq(api_key=api_key)
        
        print("🔄 Testing API call...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Updated model
            messages=[{"role": "user", "content": "Say 'test successful'"}],
            max_tokens=10
        )
        
        print("✅ Groq API connection successful!")
        print(f"   Response: {response.choices[0].message.content}")
        
        return True
    except Exception as e:
        print(f"❌ Groq API error: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n🧪 RAG System Test Suite")
    print("="*50)
    
    results = {
        "Packages": test_imports(),
        "Environment": test_env_file(),
        "Directories": test_directories(),
        "Embedding Model": test_embedding_model(),
        "Groq API": test_groq_connection()
    }
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. cd backend && python main.py")
        print("2. In another terminal: cd frontend && streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install packages: pip install -r requirements.txt")
        print("- Create .env: cp .env.example .env")
        print("- Add Groq API key to .env")
    print("="*50)


if __name__ == "__main__":
    main()
