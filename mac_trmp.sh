./rtmp/mac/ffmpeg \
-itsoffset 0.15 \
-f avfoundation \
-rtbufsize 256M \
-framerate 59.94 \
-i "2:1" \
-filter_complex "[0:v]fps=60,scale=iw:trunc(ih/2)*2,format=yuv420p[v];[0:a]aresample=async=1[a]" \
-map "[v]" \
-map '[a]' \
-c:v libx264 \
-bf 0 \
-g 60 \
-r 60 \
-crf 20 \
-c:a aac \
-ar 44100 \
-b:a 192k \
-f flv \
rtmp://localhost:1935/live

./rtmp/mac/ffmpeg \
-itsoffset 0.15 \
-f avfoundation \
-rtbufsize 256M \
-framerate 59.94 \
-i "2:1" \
-filter_complex "[0:v]fps=60,scale=iw:trunc(ih/2)*2,format=yuv420p[v];[0:a]aresample=async=1[a]" \
-map "[v]" \
-map '[a]' \
-c:v libx264 \
-bf 0 \
-g 60 \
-r 60 \
-crf 20 \
-c:a aac \
-ar 44100 \
-b:a 64k \
-f rtsp \
-rtsp_transport tcp \
rtsp://localhost:8554/live

./rtmp/mac/ffmpeg \
-itsoffset 0.15 \
-f avfoundation \
-rtbufsize 256M \
-framerate 59.94 \
-i "2:1" \
-filter_complex "[0:v]fps=60,crop=1280:720:100:50,scale=iw:trunc(ih/2)*2,format=yuv420p[v];[0:a]aresample=async=1[a]" \
-map "[v]" \
-map '[a]' \
-c:v libx264 \
-bf 0 \
-g 60 \
-r 60 \
-preset ultrafast \
-crf 20 \
-c:a libopus \
-ar 48000 \
-b:a 128k \
-f rtsp \
rtsp://localhost:8554/live

