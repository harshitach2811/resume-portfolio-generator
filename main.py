import json
import os
import sys
import time
import threading
import webbrowser
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

from dotenv import load_dotenv
from google import genai
from google.genai import types
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

# 1. PROJECT FILE PATHS

BASE_DIR = Path(__file__).resolve().parent

RESUME_FILE = BASE_DIR / "resume.txt"
TEMPLATE_FILE = BASE_DIR / "template.html"
OUTPUT_FILE = BASE_DIR / "portfolio.html"

# 2. PYDANTIC DATA MODELS

class SkillCategory(BaseModel):
    category_name: str = ""
    skills: list[str] = Field(default_factory=list)


class Project(BaseModel):
    title: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    link: str = ""


class Experience(BaseModel):
    role: str = ""
    company: str = ""
    dates: str = ""
    responsibilities: list[str] = Field(default_factory=list)


class Education(BaseModel):
    institution: str = ""
    degree: str = ""
    year: str = ""
    details: str = ""


class Certification(BaseModel):
    title: str = ""
    issuer: str = ""
    year: str = ""


class Achievement(BaseModel):
    title: str = ""
    description: str = ""


class Contact(BaseModel):
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""


class PortfolioSchema(BaseModel):
    name: str = ""
    role_badge: str = ""
    tagline: str = ""
    about_bio: str = ""

    skill_categories: list[SkillCategory] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    achievements: list[Achievement] = Field(default_factory=list)
    contact: Contact = Field(default_factory=Contact)

# 3. VALIDATE RESUME

def clean_resume_text(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def validate_resume(text: str) -> str:
    if not RESUME_FILE.exists():
        raise FileNotFoundError(
            "resume.txt was not found. Please create resume.txt in the project folder."
        )

    if not text.strip():
        raise ValueError("resume.txt is empty. Please add resume information.")

    cleaned_text = clean_resume_text(text)

    if len(cleaned_text) < 30:
        raise ValueError("resume.txt is too short. Please provide a valid resume.")

    return cleaned_text

# 4. SEND RESUME TO GEMINI (WITH FALLBACK & RETRIES)

def process_resume_with_gemini(resume_text: str) -> PortfolioSchema:
    load_dotenv(BASE_DIR / ".env", override=True)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Please add your Gemini API key to .env."
        )

    # Primary model configured in env, followed by fallbacks in order of preference
    env_model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    fallback_models = [env_model, "gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]
    
    # Deduplicate list while preserving order
    models_to_try = []
    for m in fallback_models:
        if m not in models_to_try:
            models_to_try.append(m)

    client = genai.Client(api_key=api_key)

    prompt = f"""
Read the resume below and extract portfolio details matching the required JSON structure.
Only use factual details directly present in the resume text.

RESUME CONTENT:
{resume_text}
"""

    last_exception = None

    for model_name in models_to_try:
        print(f"Attempting API request with model: {model_name}")
        
        # Try each model up to 3 times on temporary server load (503)
        for attempt in range(1, 4):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=PortfolioSchema,
                    ),
                )
                
                if response.text:
                    print(f"Success! Model '{model_name}' processed the request.")
                    return PortfolioSchema.model_validate_json(response.text)

            except Exception as exc:
                err_msg = str(exc)
                last_exception = exc

                # If 503 (Server busy), wait and retry same model
                if "503" in err_msg or "UNAVAILABLE" in err_msg:
                    print(f"  [503 High Demand] Attempt {attempt}/3 failed. Retrying in {attempt * 2}s...")
                    time.sleep(attempt * 2)
                    continue
                
                # If 404 (Model not available), skip directly to next fallback model
                if "404" in err_msg or "NOT_FOUND" in err_msg:
                    print(f"  [404 Not Found] Model '{model_name}' is not available. Switching fallback...")
                    break
                
                # Other errors (e.g. rate limit, authorization)
                print(f"  Error on '{model_name}': {err_msg}")
                break

    raise RuntimeError(
        f"All attempted Gemini models failed. Last error: {last_exception}"
    ) from last_exception

# 5. GENERATE portfolio.html

def generate_portfolio() -> Path:
    if not RESUME_FILE.exists():
        raise FileNotFoundError("resume.txt is missing.")

    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError("template.html is missing.")

    resume_text = RESUME_FILE.read_text(encoding="utf-8")
    cleaned_resume = validate_resume(resume_text)

    print("Resume validated successfully.")
    print("Sending resume information to Gemini...")

    portfolio_data = process_resume_with_gemini(cleaned_resume)

    print("Gemini returned valid portfolio data.")

    environment = Environment(
        loader=FileSystemLoader(str(BASE_DIR)),
        autoescape=select_autoescape(["html", "xml"])
    )

    template = environment.get_template("template.html")
    data = portfolio_data.model_dump()
    final_html = template.render(data=data)

    OUTPUT_FILE.write_text(final_html, encoding="utf-8")
    return OUTPUT_FILE

# 6. LOCAL WEBSITE SERVER

class PortfolioHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def start_server():
    os.chdir(BASE_DIR)

    server = ThreadingHTTPServer(("127.0.0.1", 8000), PortfolioHandler)
    url = "http://127.0.0.1:8000/portfolio.html"

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()

# 7. MAIN PROGRAM

if __name__ == "__main__":
    print()
    print("AI-Assisted Resume Portfolio Generator")
    print("--------------------------------------")

    try:
        output_file = generate_portfolio()

        print()
        print(f"Success! Generated: {output_file.name}")
        print("Open this file in your browser:")
        print("http://127.0.0.1:8000/portfolio.html")
        print()

        start_server()

    except Exception as exc:
        print()
        print(f"Error: {exc}")
        print()
        sys.exit(1)