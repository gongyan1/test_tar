'''
1.	输入：图像尺寸（m, n）,点云强度矩阵[4, N]
    # 点云格式如群里所说的，且坐标全部位于图像范围内
2.	创建矩阵[m, n], 字典dict{}
3.	for x in (0, m-1):
		for y in (0, n-1):
			计算点云中距离(x, y)最近的3个点的坐标a,b,c,以及对应的距离da,db,dc
			按距离计算权重：da' = da/(da+db+dc), db' = db/(da+db+dc), dc' = dc/(da+db+dc)
			存到字典{(x, y):a,b,c,da',db',dc'}
4. 	for x in (0, m-1):
		for y in (0, n-1):
			if(x,y) not in 点云坐标：
				(x, y) = da'*A + db'*B + dc'*C # ABC是三个坐标上点云的强度值
5. return [m, n]
'''

import kdtree

def create_dict(point_cloud):
	"""image_size: 图像尺寸（m, n）
	   point_cloud: 点云强度矩阵
	"""
    
    # 点云转强度图
    depth_image = pc2image(point_cloud)
    
	 w, h = size(image)
	
	 width = image_size[0]
	 height = image_size[1]
    
    # 将点云投影到二维平面
    point_cloud = pc2(point_cloud)
    
    # 构造kdtree
    KDTree = kdtree.create(point_cloud)
	
	 dicti = {}
	
    # 
	 for x in range(width):
		 for y in range(height):
		    _, a, b, c, da, db, dc = KDTree.search_knn(point_cloud[x, y], 4)
			 d_s = da + db + dc
			 wa, wb, wc = da/ds, db/ds, dc/ds
			
			 dicti[tuple(x, y)] = [a, b, c, wa, wb, wc]
	 
    # 补全点云强度图的空缺点
    for x in (0, m-1):
        for y in (0, n-1):
            if(x,y) == (0,0):
                a, b, c, da_, db_, dc_ = dicti[(x,y)]
                A = point_cloud[a[0]][a[1]]
                B = point_cloud[b[0]][b[1]]
                C = point_cloud[c[0]][c[1]]
			      point_cloud[x, y] = da_*A + db_*B + dc_ *C # ABC是三个坐标上点云的强度值
    
    return point_cloud
			
def calculate_intensity(dicti, )
	""" """
    for x in (0, m-1):
        for y in (0, n-1):
            if(x,y) == (0,0):
			      (x, y) = da'*A + db'*B + dc'*C # ABC是三个坐标上点云的强度值
                  

import kitti-tools-master.
if __name__ = '__main__':
    
    filename = "um_000000"
    pc_path = "./data/bin/"+filename+".bin"
    calib_path = "./data/calib/"+filename+".txt"
    image_path = "./data/img/"+filename+".png"
    
    param = data_provider.read_calib(calib_path, [2,4,5])
    
    
    
    # 读取二进制点云文件
    lidar = np.fromfile(pc_path, dtype=np.float32,count=-1).reshape([-1,4])
    
    cam2img = param[0].reshape([3,4])   # from camera-view to pixels
    cam2cam = param[1].reshape([3,3])   # rectify camera-view
    vel2cam = param[2].reshape([3,4])   # from lidar-view to camera-view

    HRES = config.HRES          # horizontal resolution (assuming 20Hz setting)
    VRES = config.VRES          # vertical res
    VFOV = config.VFOV          # Field of view (-ve, +ve) along vertical axis
    Y_FUDGE = config.Y_FUDGE    # y fudge factor for velodyne HDL 64E
    
    # 获取强度图
    lidar_to_2d_front_view(lidar, v_res=VRES, h_res=HRES, v_fov=VFOV, \
        val="reflectance", saveto=filename+"_reflectance.png", y_fudge=Y_FUDGE)
    
    image.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
			
			