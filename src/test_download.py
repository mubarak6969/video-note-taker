# Manual smoke-test script (needs network access) - not run by pytest.
from downloader import download_audio

url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

video_id, title, file_path = download_audio(url)
print(f"Video ID: {video_id}")
print(f"Title: {title}")
print(f"File Path: {file_path}")