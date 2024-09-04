import numpy as np 

# def get_videoid(video_url:str):
#     res = video_url.split("/watch?v=")
#     if len(res) == 1:
#         return np.NaN 
#     else:
#         return res[1]
    
def get_videoid(video_url:str):
    if str(video_url) != "nan":
        res = video_url.split("/watch?v=")
        if len(res) > 1:
            res = res[1].split("&amp;lc=")[0]
        else:
            res = np.NaN
    else:
        res = np.NaN
    return res