import asyncio
import numpy as np
import av
from aiohttp import web
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCSessionDescription
import queue
class SBSVideoTrack(VideoStreamTrack):
    """
    Video track for streaming SBS frames from a queue.
    Always displays the latest available frame, avoids flickering black frames.
    """
    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # initial black frame

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        # Try to get the newest frame from the queue
        while True:
            try:
                self.latest_frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break  # no more frames in queue, keep latest_frame

        frame = av.VideoFrame.from_ndarray(self.latest_frame, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

class WebRTCStreamer:
    """
    WebRTC streamer using aiohttp to serve HTML + SDP endpoints.
    """
    def __init__(self, frame_queue, host="0.0.0.0", port=8080):
        self.frame_queue = frame_queue
        self.host = host
        self.port = port
        self.pcs = set()
        self.app = web.Application()
        self.app.router.add_get("/", self.index)
        self.app.router.add_post("/offer", self.offer)

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
                const pc = new RTCPeerConnection();
                const video = document.getElementById("video");

                pc.ontrack = (event) => { video.srcObject = event.streams[0]; };

                async function startStream() {
                    const offer = await pc.createOffer({ offerToReceiveVideo: true, offerToReceiveAudio: false });
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

        # 1️⃣ Add video track BEFORE setting remote description
        pc.addTrack(SBSVideoTrack(self.frame_queue))

        # 2️⃣ Set remote description from browser offer
        await pc.setRemoteDescription(offer)

        # 3️⃣ Create and set local SDP answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        )

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