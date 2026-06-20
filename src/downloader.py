import yt_dlp
import os

def download_audio(youtube_url):
    print(f"Downloading audio from: {youtube_url}")
    
    output_path = os.path.join("downloads", "%(title)s.%(ext)s")
    
    options = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
    }
    
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        title = info.get('title', 'Unknown')
        
        # Build the actual saved file path
        safe_title = ydl.prepare_filename(info)
        file_path = os.path.splitext(safe_title)[0] + ".mp3"
        
        print(f"✅ Downloaded: {title}")
        print(f"📁 Saved at: {file_path}")
        return title, file_path