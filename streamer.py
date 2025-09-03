import asyncio
import queue
import numpy as np
import av
from aiohttp import web
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCSessionDescription, RTCIceServer

class SBSVideoTrack(VideoStreamTrack):
    def __init__(self, frame_queue, fps):
        super().__init__()
        self.frame_queue = frame_queue
        # Initialize with higher resolution black frame
        self.latest_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.last_pts = 0
        self.frame_count = 0
        self.fps = fps

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        
        # Smooth frame timing
        if self.frame_count == 0:
            self.last_pts = pts
        
        # Get the newest frame without blocking
        try:
            self.latest_frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass  # Keep previous frame if no new one available

        # Create video frame with proper timing
        frame = av.VideoFrame.from_ndarray(self.latest_frame, format="rgb24")
        frame.pts = self.last_pts
        frame.time_base = time_base
        
        # Increment pts by frame duration (assuming 30fps)
        self.last_pts += int(90000 / self.fps)  # 90000 is the clock rate for video
        self.frame_count += 1
        
        return frame

class WebRTCStreamer:
    """
    WebRTC streamer using aiohttp to serve HTML + SDP endpoints.
    """
    def __init__(self, frame_queue, host="0.0.0.0", port=8080, fps=60):
        self.frame_queue = frame_queue
        self.host = host
        self.port = port
        self.pcs = set()
        self.app = web.Application()
        self.app.router.add_get("/", self.index)
        self.app.router.add_post("/offer", self.offer)
        self.fps = fps

    async def index(self, request):
        """Serve the HTML page with only the video element filling the browser."""
        content = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                html, body {
                    margin: 0;
                    padding: 0;
                    background: black;
                    height: 100%;
                    width: 100%;
                }
                video {
                    display: block;
                    width: 100%;
                    height: auto;
                    max-height: 100%;
                }
            </style>
        </head>
        <body>
            <video id="video" autoplay playsinline controls></video>
            <script>
            const pc = new RTCPeerConnection({
                iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
                bundlePolicy: "max-bundle",
                rtcpMuxPolicy: "require"
            });
            
            // Enable hardware acceleration if available
            const video = document.getElementById("video");
            video.playsInline = true;
            video.setAttribute("webkit-playsinline", "");
            
            // Configure video constraints for better quality
            const mediaConstraints = {
                offerToReceiveVideo: true,
                offerToReceiveAudio: false,
                iceRestart: false
            };
            
            pc.ontrack = (event) => {
                if (event.track.kind === "video") {
                    video.srcObject = event.streams[0];
                    video.onloadedmetadata = () => video.play();
                }
            };
            
            async function startStream() {
                try {
                    const offer = await pc.createOffer(mediaConstraints);
                    await pc.setLocalDescription(offer);
                    
                    const resp = await fetch("/offer", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            sdp: pc.localDescription.sdp,
                            type: pc.localDescription.type
                        })
                    });
                    
                    const answer = await resp.json();
                    await pc.setRemoteDescription(answer);
                } catch (err) {
                    console.error("Streaming error:", err);
                }
            }
            
            startStream();
        </script>
        </body>
        </html>
        """
        return web.Response(content_type="text/html", text=content)

    async def offer(self, request):
        """
        Handle POSTed SDP offer from browser and return SDP answer.
        """
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        track = SBSVideoTrack(self.frame_queue, fps=self.fps)
        pc.addTrack(track)

        # Set remote description from browser offer
        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
    
        # Enhanced SDP modification for better quality
        answer.sdp = self.optimize_sdp(answer.sdp)
        
        await pc.setLocalDescription(answer)
        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    
    def optimize_sdp(self, sdp):
        """Optimize SDP for better video quality"""
        lines = sdp.split('\n')
        new_lines = []
        
        # Set higher bitrate (8000 kbps)
        bitrate = 8000
        
        # Add video quality parameters
        for line in lines:
            if line.startswith('a=mid:video'):
                new_lines.append(line)
                new_lines.append('b=AS:' + str(bitrate))
            elif line.startswith('a=rtpmap:') and 'H264' in line:
                # Add H.264 parameters for better quality
                new_lines.append(line)
                new_lines.append('a=fmtp:96 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f')
            else:
                new_lines.append(line)
        
        return '\n'.join(new_lines)
    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        print(f"[WebRTC] Streaming server running at http://{self.host}:{self.port}")
        # Keep running indefinitely
        while True:
            await asyncio.sleep(3600)

    async def stop(self):
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        print("[WebRTC] Streamer stopped")