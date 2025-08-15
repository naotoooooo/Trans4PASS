import os
import glob
import numpy as np
import cv2

def birds_eye_equirectangular_corrected(equi_img,
                                        camera_height=1.5,
                                        grid_forward=10.0,
                                        grid_side=10.0,
                                        resolution=0.05):
    """
    歪みを考慮してエクイレクタングラー画像から鳥瞰図を生成する
    """
    H, W = equi_img.shape[:2]  
    print(f" H: {H}, W: {W}")  
    expected_H = int(W / 2)  # 2:1画像の期待される高さ
    H_start = (expected_H - H) // 2  # ← まずはこれでOK
    # H_start = 145  # ← まずはこれでOK
    print(f" H_start: {H_start}")  # トリミング開始位置を確認

    # 出力鳥瞰画像のサイズ
    out_H = int(grid_forward / resolution)
    out_W = int((2 * grid_side) / resolution)
    print(f" out_H: {out_H}, out_W: {out_W}")  # 出力画像のサイズを確認

    # 地面のXY座標グリッドを定義（カメラ下中心）
    x_vals = np.linspace(-grid_side, grid_side, out_W)
    # y_vals = np.linspace(0, grid_forward, out_H)
    y_vals = np.linspace(-grid_forward/2, grid_forward/2, out_H)
    
    
    #     # y_vals を非線形スケーリングで中央密度を高く
    # linear = np.linspace(-1, 1, out_H)
    # y_vals = np.sign(linear) * (np.abs(linear) ** 0.7)  # √で密度強調
    # y_vals = (y_vals + 1) / 2  # [0,1]
    # y_vals = y_vals * grid_forward - (grid_forward / 2)

    
    
    
    
    xv, yv = np.meshgrid(x_vals, y_vals)

    # 各地面点からカメラ中心への方向ベクトル（カメラは z = camera_height にある）
    dx = xv
    dy = yv
    dz = -camera_height * np.ones_like(dx)

    # 方向ベクトルを正規化（球面上の点）
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= norm
    dy /= norm
    dz /= norm

    # 方位角（longitude）と仰角（latitude）
    lon = np.arctan2(dy, dx)  # [-π, π]
    lat = np.arcsin(dz)       # [-π/2, π/2]
    # print(f" lon: {lon}, lat: {lat}")  # lonとlatの形状を確認

    # エクイレクタングラー画像上のUV座標に変換（歪み考慮）
    # u = (lon / (2 * np.pi) + 0.5) * W
    # v = (0.5 - lat / np.pi) * H
    # print(f" u: {u}, v: {v}")  # uとvの形状を確認
    
    
    
    
    # # 例えば、上10%と下10%がトリミングされていると仮定して
    # visible_lat_min = -np.pi/2 * 1.0 # 本来 -π/2 → 実質この範囲
    # visible_lat_max = +np.pi/2 * 1.0
    # v = ((visible_lat_max - lat) / (visible_lat_max - visible_lat_min)) * H



    # # # v = (0.5 - lat / np.pi) * H # --- 重要部分：vを元の expected_H に合わせてマッピング ---
    # v = (0.5 - lat / np.pi) * expected_H - 145# 本来の2:1画像基準のv
    v = (0.5 - lat / np.pi) * expected_H - 56  # 本来の2:1画像基準のv

    # # # しかし画像は H < expected_H → 範囲外を黒くする（cv2が自動でやる）
    # # v_full_clipped = v_full.copy()

    # # # 範囲外は -1 にして remap で黒く塗らせる
    # # v_full_clipped[(v_full < Hstart) | (v_full >= H+H_start)] = -1

    # # 横方向uも通常通り
    u = (lon / (2 * np.pi) + 0.5) * W
    # # u = np.clip(u, 0, W - 1)  # u方向はトリミングされていないと仮定
    u = np.clip(u, 0, W - 1)  # u方向はトリミングされていないと仮定




    # OpenCV remap用 float32マップ
    map_x = u.astype(np.float32) ###問題あり？
    map_y = v.astype(np.float32)
    
    

    # 出力鳥瞰ビュー画像
    # birds_eye = cv2.remap(equi_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    birds_eye = cv2.remap(equi_img, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))   ##色がグラデーションにならない
    # birds_eye = cv2.flip(birds_eye, 1)  # 水平方向（左右）を反転

    
    # 出力を180度回転（上下・左右反転）
    birds_eye = cv2.rotate(birds_eye, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return birds_eye


# # img = cv2.imread("results_8noji_test/1740579295.965225.png_seg_only.png")
# # img = cv2.imread("results_8noji_test/1740578825.992192.png_seg_only.png")
# img = cv2.imread("datasets/outdoor_8noji_image_raw_2025-02-26-14-06-25/rgb/val/1740578825.992192.png")
# # img = cv2.imread("datasets/outdoor_8noji_image_raw_2025-02-26-14-06-25/rgb/val/1740579295.965225.png")
# # img = cv2.imread("results_8noji_test/1740579289.186688.png_seg_only.png")


# bev = birds_eye_equirectangular_corrected(img,
#                                           camera_height=1.45,
#                                           grid_forward=40.0,
#                                           grid_side=20.0,
#                                           resolution=0.05)

# # cv2.imshow("Corrected Bird's Eye View", bev)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # 保存
# cv2.imwrite("birds_eye_output_imgxxx.png", bev)

input_dir = "results_outdoor_sii_intersection2"
output_dir = "bev_results_outdoor_sii_intersection2"
os.makedirs(output_dir, exist_ok=True)

# 画像ファイル取得
image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))

# すべての画像を処理
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"読み込み失敗: {path}")
        continue


    bev = birds_eye_equirectangular_corrected(img,
                                            camera_height=1.46,
                                            grid_forward=30.0,
                                            grid_side=15,
                                            resolution=0.05)

    # cv2.imshow("Corrected Bird's Eye View", bev)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存
# cv2.imwrite("birds_eye_output_imgxxx4.png", bev)
    basename = os.path.basename(path)
    out_path = os.path.join(output_dir, f"bev_{basename}")
    cv2.imwrite(out_path, bev)