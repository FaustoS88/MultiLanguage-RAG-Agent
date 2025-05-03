import subprocess

def run(cmd):
    print(f"> {cmd}")
    subprocess.check_call(cmd, shell=True)

if __name__ == "__main__":
    steps = [
        # clean up everything stale
        "docker compose down --volumes --remove-orphans",

        # build and start DB
        "docker compose build",
        "docker compose up -d db",

        # init schema + extension + table
        "docker compose run --rm init-db",

        # crawl docs (fills context7_docs)
        "docker compose run --rm crawler",

        # finally launch the app
        "docker compose up -d app",
        # final step: stream logs continuously
        "docker compose logs -f app"
    ]
    for cmd in steps:
        run(cmd)
    print("\n✅ All set! Streamlit RAG app is live at http://localhost:8501")
