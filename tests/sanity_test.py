import os


def test_env_loaded():
    print(os.environ)
    assert "CONNECTION_URL" in os.environ
    assert "MONGO_DB" in os.environ
