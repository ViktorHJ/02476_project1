import os
import typer

app = typer.Typer()

@app.command()
def create_req_files():
    os.system("uv export --format requirements.txt --no-dev > requirements.txt")
    os.system("uv export --format requirements.txt --dev > requirements_dev.txt")
    print("Requirement files created: requirements.txt, requirements_dev.txt")
    
if __name__ == "__main__":
    app()