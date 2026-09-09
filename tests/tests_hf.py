import requests

from huggingface_hub import configure_http_backend, get_session, hf_hub_download


def backend_factory():
    session = requests.Session()
    session.verify = False
    return session


configure_http_backend(backend_factory)

print("Session verify setting:", get_session().verify)

path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="modules.json",
)

print("Downloaded successfully:")
print(path)