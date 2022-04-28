import h5py
import struct
import zlib
import numpy as np

from pytorch3d.io import load_obj

def load_BFM_2019(fname="./model2017-1_bfm_nomouth.h5"):

    with h5py.File(fname, 'r') as f:
        shape_mean = f['shape']['model']['mean'][:]
        shape_pcaBasis = f['shape']['model']['pcaBasis'][:]
        shape_pcaVariance = f['shape']['model']['pcaVariance'][:]

        expression_mean = f['expression']['model']['mean'][:]
        expression_pcaBasis = f['expression']['model']['pcaBasis'][:]
        expression_pcaVariance = f['expression']['model']['pcaVariance'][:]

        color_mean = f['color']['model']['mean'][:]
        color_pcaBasis = f['color']['model']['pcaBasis'][:]
        color_pcaVariance = f['color']['model']['pcaVariance'][:]

        faces = f['shape']['representer']['cells'][:].transpose(1,0)
        faces = torch.tensor(faces)

        print(shape_mean.shape)
        print(shape_pcaBasis.shape)
        print(shape_pcaVariance.shape)
        print(faces.shape)

        return {
            'shape_mean': shape_mean,
            'shape_pcaBasis': shape_pcaBasis, 
            'shape_pcaVariance': shape_pcaVariance,
            'expression_mean': expression_mean, 
            'expression_pcaBasis': expression_pcaBasis, 
            'expression_pcaVariance': expression_pcaVariance,
            'color_mean': color_mean, 
            'color_pcaBasis': color_pcaBasis, 
            'color_pcaVariance': color_pcaVariance,
            'face': faces}
        # return {'verts': v_bfm, 'color': c_bfm, 'shape_coeffs': shape_coeffs, 'exp_coeffs': exp_coeffs, 'color_coeffs': color_coeffs}


def load_facewarehouse_bin(bin_path, obj_path):
    with open(bin_path, "rb") as f:
        # Basic parameters
        vecNum = struct.unpack("i", f.read(4))[0]
        ptNum = struct.unpack("i", f.read(4))[0]

        # Average face
        mAvergPosList = []
        for i in range(ptNum):
            x,y,z = struct.unpack("fff", f.read(12))
            mAvergPosList.append(np.array([x,y,z]))
        mAvergPos = np.stack(mAvergPosList)
        
        # Eigen values
        mEigenValList = []
        for i in range(vecNum):
            e = struct.unpack("f", f.read(4))[0]
            mEigenValList.append(e)
        mEigenVal = np.array(mEigenValList)

        mPrinCompList = []
        for i in range(vecNum):
            prinComp = []
            for j in range(ptNum):
                x,y,z = struct.unpack("fff", f.read(12))
                prinComp.append(np.array([x,y,z]))
            prinComp = np.stack(prinComp)
            mPrinCompList.append(prinComp)
        mPrinCompList = np.array(mPrinCompList)
        mPrinCompList = mPrinCompList.reshape(199,-1).transpose(1,0)

        _, faces, _ = load_obj(obj_path)
        faces = faces.verts_idx

    return {
        'shape_mean': mAvergPos, 
        'shape_pcaBasis': mPrinCompList, 
        'shape_pcaVariance': mEigenVal,
        'face': faces}

def load_depth(depth_path, w, h):
    # Reading depth
    with open(depth_path, 'rb') as f:
        data = zlib.decompress(f.read(), -15)

    FRAME_COUNT = int(len(data) / w / h / 2)

    frames_in_meters = np.frombuffer(data, np.float16).reshape(FRAME_COUNT,h,w).copy()
    frames_in_meters = np.nan_to_num(frames_in_meters, 0)
    frames_in_mm = frames_in_meters * 1000.0

    mask = np.ones(frames_in_mm.shape)
    frames_in_mm[(frames_in_mm<100) + (frames_in_mm>450)] = 0
    mask[(frames_in_mm<100) + (frames_in_mm>450)] = 0
    imgs_depth = frames_in_mm.clip(100.0, 450.0)

    return imgs_depth, mask
