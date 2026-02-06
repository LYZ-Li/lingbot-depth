import os
import time
import zmq
import numpy as np
import torch
import cv2
import rerun as rr
import warnings

# === 禁用不必要的警告 ===
os.environ["XFORMERS_DISABLED"] = "1"
warnings.filterwarnings("ignore")

from mdm.model.v2 import MDMModel

# === 配置 ===
PORT = 5555
MODEL_PATH = "robbyant/lingbot-depth-pretrain-vitl-14" 

def main():
    print(f"📦 Rerun SDK Version: {rr.__version__}")
    
    # 1. 初始化 Web Viewer
    rr.init("LingBot_Live", spawn=True)
    
    # ✅ 你发现的正确写法
    print("🌐 启动 Web Server...")
    try:
        # 写法 A: 针对部分版本的写法
        rr.start_web_viewer_server(port=9090, host="0.0.0.0")
    except TypeError:
        try:
            # 写法 B: 针对另一部分版本的写法
            rr.start_web_viewer_server("0.0.0.0:9090")
        except TypeError:
            # 写法 C: 如果都不行，回退到默认，然后我们要用方法二 (socat)
            print("⚠️ 无法通过代码绑定 IP，将使用默认 localhost")
            rr.start_web_viewer_server()
    
    print("\n🌐 ==========================================")
    print("   请在 Windows 浏览器打开以下地址查看可视化:")
    print("   http://localhost:9090")
    print("   ==========================================\n")

    # 2. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 正在加载模型到 {device}...")
    try:
        model = MDMModel.from_pretrained(MODEL_PATH).to(device)
    except Exception as e:
        print(f"❌ 模型加载失败，尝试加载本地备份... ({e})")
        model = MDMModel.from_pretrained("ckpt/model.pt").to(device)
    
    model.eval()
    print("✅ 模型就绪!")

    # 3. ZMQ 反向连接
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    
    print(f"👂 正在监听端口 {PORT}，等待 Windows 连接...")
    socket.bind(f"tcp://0.0.0.0:{PORT}")
    
    socket.setsockopt_string(zmq.SUBSCRIBE, "") 
    socket.setsockopt(zmq.CONFLATE, 1)

    # 增加一个帧计数器，万一需要用它来做时间轴
    frame_count = 0

    try:
        while True:
            # 4. 接收数据
            header = socket.recv_json()
            depth_bytes = socket.recv()
            color_bytes = socket.recv()

            w, h = header["w"], header["h"]
            intr = header["intr"]
            
            # 5. 解码数据
            raw_depth_mm = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(h, w)
            depth_m = raw_depth_mm.astype(np.float32) / 1000.0
            
            raw_rgb_bgr = np.frombuffer(color_bytes, dtype=np.uint8).reshape(h, w, 3)
            raw_rgb = cv2.cvtColor(raw_rgb_bgr, cv2.COLOR_BGR2RGB)
            
            # 6. 推理
            img_tensor = torch.from_numpy(raw_rgb).float().to(device) / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            depth_tensor = torch.from_numpy(depth_m).float().to(device)

            K = np.array([
                [intr['fx'], 0,          intr['ppx']],
                [0,          intr['fy'], intr['ppy']],
                [0,          0,          1]
            ], dtype=np.float32)
            
            K_norm = K.copy()
            K_norm[0, 0] /= w; K_norm[0, 2] /= w
            K_norm[1, 1] /= h; K_norm[1, 2] /= h
            intrinsics_tensor = torch.from_numpy(K_norm).unsqueeze(0).to(device)

            # 计时推理耗时
            t_infer_start = time.time()
            with torch.no_grad():
                output = model.infer(
                    img_tensor, depth_in=depth_tensor, apply_mask=True, intrinsics=intrinsics_tensor
                )
            t_infer_duration = time.time() - t_infer_start
            pred_depth = output['depth'].squeeze().cpu().numpy()
            
            # ==========================================================
            # 7. Rerun 可视化 (关键修复部分)
            # ==========================================================
            
            # 修复方案：尝试设置时间轴，如果报错就跳过，不影响显示
            try:
                # 优先尝试用整数帧号，通常比 set_time_seconds 稳定
                if hasattr(rr, 'set_time_sequence'):
                    rr.set_time_sequence("frame_idx", frame_count)
                elif hasattr(rr, 'set_time_seconds'):
                    rr.set_time_seconds("capture_time", header["t"])
                else:
                    # 如果啥都没有，就什么都不做，Rerun 会自动按接收时间排序
                    pass
            except Exception:
                pass # 忽略所有时间设置错误

            rr.log("camera/rgb", rr.Image(raw_rgb))
            rr.log("camera/depth/raw", rr.DepthImage(raw_depth_mm, meter=1000))
            rr.log("camera/depth/refined", rr.DepthImage(pred_depth, meter=1)) 
            rr.log("camera", rr.Pinhole(resolution=[w, h], image_from_camera=K))

            # 打印状态
            if frame_count % 30 == 0:
                 print(f"\r🚀 Running... Infer: {t_infer_duration*1000:.1f}ms", end="")
            
            frame_count += 1

    except KeyboardInterrupt:
        print("\n🛑 退出...")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()