# ffmpeg -f x11grab -framerate 30 -window_id 0x500000b -i :1.0 \
#        -f alsa -ac 2 -i hw:0 \
#        -c:v libx264 -preset veryfast -tune zerolatency \
#        -c:a libopus -b:a 128k \
#        -f rtsp -rtsp_transport tcp rtsp://localhost:8554/live


       
ffmpeg -fflags +genpts+nobuffer -flags low_delay -probesize 64 -analyzeduration 0 \
-f x11grab -framerate 60 -window_id 0x520000b -use_wallclock_as_timestamps 1 \
-thread_queue_size 2048 -i :1.0 \
-f pulse -thread_queue_size 512 -i 1 -ac 2 -filter_complex "[0:v]fps=60,scale=iw:trunc(ih/2)*2,format=yuv420p[v];[1:a]aresample=async=1[a]" \
-map "[v]" -map "[a]" \
-c:v libx264 -preset ultrafast -tune zerolatency -r 60 \
-c:a libopus -b:a 128k -ar 48000 \
-f rtsp rtsp://localhost:8554/live