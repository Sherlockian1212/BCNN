# import SimpleITK as sitk
# import matplotlib.pylab as plt
#
# ct_scans = sitk.GetArrayFromImage(sitk.ReadImage(r"E:\1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd", sitk.sitkFloat32))
# plt.figure(figsize=(20,16))
# plt.gray()
# plt.subplots_adjust(0,0,1,1,0.01,0.01)
#
# max_slices = min(ct_scans.shape[0], 30)  # Giới hạn tối đa 30 lát cắt
# for i in range(max_slices):
#     plt.subplot(5, 6, i+1)
#     plt.imshow(ct_scans[i])
#     # use plt.savefig(...) here if you want to save the images as .jpg, e.g.,
#     plt.axis('off')
#
# plt.show()




# import pandas as pd
#
# # Đọc file CSV
# df = pd.read_csv(r"E:\DATA_Lung\subset0.csv")
#
# total_rows = len(df)
#
# print(f"Tổng số dòng trong file: {total_rows}")
#
# # Đếm số lượng người duy nhất dựa trên cột seriesuid
# num_people = df['seriesuid'].nunique()
#
# print(f"Số lượng người trong file: {num_people}")
#
# counts = df.groupby('seriesuid').size()
#
# # Lấy số nhỏ nhất và số lớn nhất
# min_count = counts.min()
# max_count = counts.max()
#
# print(f"Ít nhất: {min_count} ảnh cho 1 người")
# print(f"Nhiều nhất: {max_count} ảnh cho 1 người")



# import pandas as pd
# import os
#
# # Đường dẫn đến thư mục chứa các file .raw
# folder_path = "E:\DATA_Lung\subset0\subset0"  # Thay bằng đường dẫn của bạn
#
# # Đọc file CSV gốc
# csv_file = "E:\DATA_Lung\candidates_V2.csv"  # Thay bằng đường dẫn file CSV của bạn
# df = pd.read_csv(csv_file)
#
# # Lấy danh sách các tên từ file .raw (không bao gồm phần mở rộng)
# raw_names = [os.path.splitext(file)[0] for file in os.listdir(folder_path) if file.endswith('.raw')]
#
# # Lọc các dòng trong file CSV có seriesuid trùng với tên file .raw
# filtered_df = df[df['seriesuid'].isin(raw_names)]
#
# # Ghi các dòng trùng vào file CSV mới
# output_csv = r'E:\DATA_Lung\filtered_file.csv'
# filtered_df.to_csv(output_csv, index=False)
#
# print(f"Đã lưu các dòng trùng vào: {output_csv}")


# import pandas as pd
# import os
# import shutil
#
# # Đường dẫn các thư mục
# csv_file = r"E:\DATA_Lung\subset0.csv"  # Thay bằng đường dẫn file CSV
# folder_A = r"E:\DATA_Lung\subset0\subset0"  # Thư mục nguồn chứa các file
# folder_B = r"E:\DATA_Lung\Nodules_filter"  # Thư mục đích để lưu file
#
# # Đọc file CSV
# df = pd.read_csv(csv_file)
#
#
# ct_scans = sitk.GetArrayFromImage(sitk.ReadImage(r"E:\.mhd", sitk.sitkFloat32))
#
#
# # Lưu file CSV đã lọc
# output_csv = r'E:\DATA_Lung\filtered_subset0.csv'
#
# print(f"Đã lưu file CSV lọc tại: {output_csv}")

#
# import SimpleITK as sitk
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def world_to_voxel(coord, origin, spacing):
#     """
#     Chuyển từ tọa độ thế giới (X, Y, Z) sang tọa độ voxel (Z, Y, X)
#     """
#     # Đảo thứ tự trục thành (Z, Y, X)
#     coord = np.array(coord)[::-1]
#     origin = np.array(origin)
#     spacing = np.array(spacing)
#
#     voxel = (coord - origin) / spacing
#     return np.round(voxel).astype(int)
#
#
# def extract_image(mhd_file, coord):
#     # Đọc file .mhd
#     itk_image = sitk.ReadImage(mhd_file)
#     image_array = sitk.GetArrayFromImage(itk_image)  # (Z, Y, X)
#
#     origin = np.array(itk_image.GetOrigin())  # (X, Y, Z)
#     spacing = np.array(itk_image.GetSpacing())  # (X, Y, Z)
#
#     # Chuyển từ tọa độ thế giới sang tọa độ voxel
#     voxel_coord = world_to_voxel(coord, origin, spacing)
#
#     # Lấy chỉ số lát cắt Z
#     z = voxel_coord[0]
#     y, x = voxel_coord[1], voxel_coord[2]
#
#     # Kiểm tra chỉ số Z hợp lệ
#     if z < 0 or z >= image_array.shape[0]:
#         print("Lỗi: Chỉ số lát cắt ngoài phạm vi.")
#         return
#
#     # Hiển thị ảnh lát cắt
#     plt.imshow(image_array[z], cmap='gray')
#     plt.title(f'Z-slice at {z}')
#     plt.scatter(x, y, color='red', marker='x')
#     plt.axis('off')
#     plt.show()
#
#
# # Ví dụ sử dụng
# mhd_file = r"E:\DATA_Lung\subset0\subset0\1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd"
# coord = [-128.6994211, -175.3192718, -298.3875064]  # (X, Y, Z)
# extract_image(mhd_file, coord)
