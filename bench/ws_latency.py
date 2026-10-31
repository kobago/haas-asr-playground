"""Stream an audio file to the playground's realtime WebSocket in real time.

python ws_latency.py "wss://HOST:7860/ws/realtime?model=onnx-q" audio.flac

Sends 100 ms int16 frames (each prefixed with a float64 timestamp in ms, as the
browser does) and prints the delay of every update: the time from sending the
newest frame in that update to receiving the update. Ends with the settled phrases.
"""
import asyncio, json, struct, sys, time
import numpy as np, soundfile as sf, websockets
async def main(url, path):
    x, _ = sf.read(path, dtype="int16")
    x = np.concatenate([x, np.zeros(16000 * 2, dtype=np.int16)])  # trailing silence to settle
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "start"}))
        t0 = time.perf_counter(); updates = 0; last = None
        async def reader():
            nonlocal updates, last
            async for m in ws:
                msg = json.loads(m); updates += 1; last = msg if msg["type"] == "update" else last
                if msg["type"] == "update" and msg.get("latency"):
                    d = (time.perf_counter() * 1000 - msg["latency"]["stamp"])
                    print(f"t={time.perf_counter()-t0:4.1f}s delay={d:5.0f}ms model={msg['latency']['model_ms']:4.0f}ms pending={msg['pending']['text'] if msg['pending'] else ''}")
                if msg["type"] in ("stopped", "error"): print(msg); return
        task = asyncio.create_task(reader())
        for i in range(0, len(x), 1600):  # 100 ms frames in real time
            await ws.send(struct.pack("<d", time.perf_counter() * 1000) + x[i:i+1600].tobytes())
            await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "stop"}))
        await task
        print("settled:", [p["text"] for p in last.get("phrases", [])] if last else None)
asyncio.run(main(sys.argv[1], sys.argv[2]))
