import imageio # 导入imageio库
import os


idx=29

def creat_gif(image_list, gif_name, duration,root_path):
    """
    生成gif文件，原始图像仅仅支持png格式；
    gif_name : 字符串，所生成的gif文件名，带.gif文件名后缀；
    path : 输入图像的路径；
    duration : gif图像时间间隔，这里默认设置为1s,当然你喜欢可以设置其他；
    """
    # 创建一个空列表，用来存源图像
    frames = []
    
    # 利用方法append把图片挨个存进列表
    
    for image_name in image_list:
        frames.append(imageio.imread(root_path+image_name))
 
    # 保存为gif格式的图
    imageio.mimsave(gif_name, frames, 'GIF', duration = duration)
 
    return
    

name='pic2'
#root_path='C:\\Users\\35968\\Desktop\\temp\\'+name+'\\'
root_path='E:\\gray_smoke_256\\2\\pic\\'
image_list= os.listdir(root_path)
image_list.sort(key= lambda x:int(x[:-4]))
gif_name = name+'.gif'
duration = 0.1
creat_gif(image_list, gif_name,duration,root_path)



















