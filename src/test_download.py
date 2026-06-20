from downloader import download_audio

url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

title, file_path = download_audio(url)
print(f"Title: {title}")
print(f"File Path: {file_path}")