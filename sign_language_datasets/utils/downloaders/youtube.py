def download_youtube(url, dst_path):
    import youtube_dl  # Required for YouTube downloads

    ydl_opts = {"format": "bestvideo", "outtmpl": dst_path}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
        except youtube_dl.utils.DownloadError as e:
            print("Problem downloading youtube video", url)
