import cv2

def get_camera_properties(camera_index=0):
    cap = cv2.VideoCapture(camera_index) # 0 表示默认摄像头，如果有多摄像头可以尝试 1, 2 等
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_index}")
        return

    print(f"--- 摄像头 {camera_index} 支持的参数 ---")

    # 获取当前分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"当前分辨率: {width}x{height}")

    # 获取当前帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"当前帧率 (FPS): {fps}")

    # 获取当前亮度 (如果支持)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    print(f"当前亮度: {brightness}")

    # 获取当前对比度 (如果支持)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    print(f"当前对比度: {contrast}")

    # 获取当前饱和度 (如果支持)
    saturation = cap.get(cv2.CAP_PROP_SATURATION)
    print(f"当前饱和度: {saturation}")

    # 获取当前曝光 (如果支持)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    print(f"当前曝光: {exposure}")

    # 获取当前白平衡 (如果支持)
    white_balance_blue_u = cap.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)
    print(f"当前白平衡 (蓝): {white_balance_blue_u}")
    # white_balance_red_v = cap.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V) # 某些摄像头可能支持
    # print(f"当前白平衡 (红): {white_balance_red_v}")

    # 获取焦距 (如果支持且摄像头是可变焦的)
    # OpenCV 没有直接的 CAP_PROP_FOCAL_LENGTH, 通常焦距是相机内部参数，不直接暴露给用户控制
    # 如果您的摄像头支持软件变焦，可能需要查阅其SDK或尝试其他更底层的库

    # 尝试设置分辨率 (不保证所有摄像头都支持任意分辨率设置)
    # 注意：设置分辨率需要在 cap.read() 之前
    # desired_width = 1280
    # desired_height = 720
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    # print(f"\n尝试设置分辨率为 {desired_width}x{desired_height}")
    # print(f"实际分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # 尝试设置帧率
    # desired_fps = 30
    # cap.set(cv2.CAP_PROP_FPS, desired_fps)
    # print(f"尝试设置帧率为 {desired_fps}")
    # print(f"实际帧率: {cap.get(cv2.CAP_PROP_FPS)}")

    # 尝试设置曝光 (负值表示自动曝光，正值表示手动曝光值)
    # cap.set(cv2.CAP_PROP_EXPOSURE, -6) # 自动曝光
    # cap.set(cv2.CAP_PROP_EXPOSURE, -5) # 示例手动曝光值，具体范围取决于摄像头
    # print(f"尝试设置曝光为 {cap.get(cv2.CAP_PROP_EXPOSURE)}")

    # 获取支持的分辨率列表 (OpenCV 没有直接的方法获取所有支持的分辨率)
    # 通常需要尝试设置常见分辨率来判断，或者查阅摄像头制造商的文档。
    # 某些摄像头驱动可能会在 CAP_PROP_FORMAT 或其他属性中提供 hint。
    print("\n注意：OpenCV 没有直接获取所有支持分辨率列表的 API。")
    print("通常需要通过尝试设置来验证，或查阅相机驱动/SDK文档。")

    # 读取并显示一帧画面
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Camera Feed", frame)
        cv2.waitKey(1) # 等待1毫秒，确保窗口显示

    cap.release()
    cv2.destroyAllWindows()

# 调用函数获取和设置摄像头参数
get_camera_properties(1) # 尝试获取第一个摄像头

times = []

def set_camera_fps(camera_index=0, desired_fps=120):
    cap = cv2.VideoCapture(camera_index) # 0 表示默认摄像头，如果有多摄像头可以尝试 1, 2 等

    if not cap.isOpened():
        print(f"错误：无法打开摄像头 {camera_index}")
        return

    print(f"尝试将摄像头 {camera_index} 的帧率设置为 {desired_fps} FPS...")

    # 尝试设置帧率
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    # 获取实际设置的帧率
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"实际帧率 (FPS): {actual_fps}")

    if actual_fps >= desired_fps:
        print(f"成功将帧率设置为或接近 {desired_fps} FPS。")
    else:
        print(f"未能将帧率设置为 {desired_fps} FPS。实际帧率为 {actual_fps} FPS。")
        print("这可能是因为您的摄像头或其驱动不支持所请求的帧率。")

    # 你可以继续读取帧来验证
    import time
    ret = True
    time123 =time.time()
    import numpy as np
    while (ret and len(times)<360):
        
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Camera Feed", frame)
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()
        times.append(time.time()-time123)
    print("平均耗时",np.mean(np.array(times)))
# 调用函数尝试设置第一个摄像头的帧率为 120 FPS
set_camera_fps(0, 120)

