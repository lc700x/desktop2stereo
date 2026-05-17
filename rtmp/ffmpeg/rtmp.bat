.\bin\ffmpeg.exe -filter_complex "gfxcapture=monitor_idx=0:max_framerate=60,hwdownload,format=bgra,format=yuv420p" -f dshow -i audio="立体声混音 (Realtek(R) Audio)" -vcodec libx264 -preset veryfast -tune zerolatency -acodec aac -ar 44100 -b:a 128k  -f flv rtmp://localhost:1935/hls/stream

.\bin\ffmpeg.exe -filter_complex "gfxcapture=window_title='(?i)Stereo Viewer':max_framerate=60,hwdownload,format=bgra,format=yuv420p" -f dshow -i audio="立体声混音 (Realtek(R) Audio)" -vcodec libx264 -preset veryfast -tune zerolatency -acodec aac -ar 44100 -b:a 128k  -f flv rtmp://localhost:1935/hls/stream

