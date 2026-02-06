import pyrealsense2 as rs
import numpy as np
import zmq
import time
import json
import subprocess
import re

PORT = 5555
WIDTH, HEIGHT = 640, 480
FPS = 30

def get_wsl_ip():
    """通过运行 wsl 命令获取 WSL 虚拟机的 IP 地址"""
    try:
        # 运行 wsl hostname -I 获取 IP
        result = subprocess.run(["wsl", "hostname", "-I"], capture_output=True, text=True)
        ip = result.stdout.strip().split(' ')[0]
        if ip:
            return ip
    except Exception:
        pass
    print("❌ 无法自动获取 WSL IP，请手动输入")
    return input("请输入 WSL 的 IP 地址 (在 WSL 输入 hostname -I 查看): ").strip()

def main():
    # --- 启动 RealSense (保持不变) ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    align = rs.align(rs.stream.color)
    
    print("📷 启动相机...")
    profile = pipeline.start(config)
    
    # 获取内参
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intrinsics_dict = {"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy}

    # --- ZMQ 改动部分 ---
    wsl_ip = get_wsl_ip()
    print(f"🎯 目标 WSL IP: {wsl_ip}")
    
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    
    # ⚠️ 关键改动：这里不再是 bind，而是 connect
    target_addr = f"tcp://{wsl_ip}:{PORT}"
    print(f"📡 正在尝试穿透防火墙连接到: {target_addr}")
    socket.connect(target_addr)

    frame_count = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame: continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            header = {
                "t": time.time(),
                "w": WIDTH, "h": HEIGHT,
                "seq": frame_count,
                "intr": intrinsics_dict
            }
            
            socket.send_json(header, flags=zmq.SNDMORE)
            socket.send(depth_image.tobytes(), flags=zmq.SNDMORE)
            socket.send(color_image.tobytes())

            if frame_count % 30 == 0:
                print(f"\r🚀 Sending frame {frame_count} -> WSL", end="")
            frame_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()