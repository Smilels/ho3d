"""
Visualize the projections in published HO-3D dataset
"""
from os.path import join
import pip
import argparse
from utils.vis_utils import *
import random
from IPython import embed
import trimesh
import open3d as o3d


def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        from pip._internal.main import main as pipmain
        pipmain(['install', package])

try:
    import matplotlib.pyplot as plt
except:
    install('matplotlib')
    import matplotlib.pyplot as plt

try:
    import chumpy as ch
except:
    install('chumpy')
    import chumpy as ch

try:
    import pickle
except:
    install('pickle')
    import pickle

import cv2
from mpl_toolkits.mplot3d import Axes3D

MANO_MODEL_PATH = './mano/models/MANO_RIGHT.pkl'

# mapping of joints from MANO model order to simple order(thumb to pinky finger)
jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]

if not os.path.exists(MANO_MODEL_PATH):
    raise Exception('MANO model missing! Please run setup_mano.py to setup mano folder')
else:
    from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model


def forwardKinematics(fullpose, trans, beta):
    '''
    MANO parameters --> 3D pts, mesh
    :param fullpose:
    :param trans:
    :param beta:
    :return: 3D pts of size (21,3)
    '''

    assert fullpose.shape == (48,)
    assert trans.shape == (3,)
    assert beta.shape == (10,)

    m = load_model(MANO_MODEL_PATH, ncomps=6, flat_hand_mean=True)
    m.fullpose[:] = fullpose
    m.trans[:] = trans
    m.betas[:] = beta

    return m.J_transformed.r, m


if __name__ == '__main__':
    # parse the input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("ho3d_path", type=str, help="Path to HO3D dataset")
    ap.add_argument("ycbModels_path", type=str, help="Path to ycb models directory")
    ap.add_argument("-split", required=False, type=str,
                    help="split type", choices=['train', 'evaluation'], default='train')
    ap.add_argument("-seq", required=False, type=str,
                    help="sequence name")
    ap.add_argument("-id", required=False, type=str,
                    help="image ID")
    ap.add_argument("-visType", required=False,
                    help="Type of visualization", choices=['open3d', 'matplotlib'], default='matplotlib')
    args = vars(ap.parse_args())

    baseDir = args['ho3d_path']
    YCBModelsDir = args['ycbModels_path']
    split = args['split']

    # some checks to decide if visualizing one single image or randomly picked images
    if args['seq'] is None:
        args['seq'] = random.choice(os.listdir(join(baseDir, split)))
        runLoop = True
    else:
        runLoop = False

    if args['id'] is None:
        args['id'] = random.choice(os.listdir(join(baseDir, split, args['seq'], 'rgb'))).split('.')[0]
    else:
        pass

    while(True):
        seqName = args['seq']
        id = args['id']
        print(seqName, id)

        # read image, depths maps and annotations
        img = read_RGB_img(baseDir, seqName, id, split)
        depth = read_depth_img(baseDir, seqName, id, split)
        anno = read_annotation(baseDir, seqName, id, split)

        # get object 3D corner locations for the current pose
        objCorners = anno['objCorners3DRest']
        objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']

        # get the hand Mesh from MANO model for the current pose
        if split == 'train':
            handJoints3D, handMesh = forwardKinematics(anno['handPose'], anno['handTrans'], anno['handBeta'])

        # fingertip 3d keypoints
        tipJoints3D = handJoints3D[16:]
        # vertices of hand area
        tip_v_dict = {}
        if hasattr(handMesh, 'r'):
            th_tip_v = np.vstack([handMesh.r[717:719, :], handMesh.r[730:731, :], handMesh.r[733:741, :],
                                  handMesh.r[743:747, :], handMesh.r[748:752, :], handMesh.r[756, :],
                                  handMesh.r[759:769, :]])  # 30
            tip_v_dict[0] = th_tip_v
            ff_tip_v = np.vstack([handMesh.r[317:321, :], handMesh.r[322:330, :], handMesh.r[332:334, :],
                                  handMesh.r[338:340, :], handMesh.r[343, :], handMesh.r[346:356, :]])  # 27
            tip_v_dict[1] = ff_tip_v
            mf_tip_v = np.vstack([handMesh.r[450, :], handMesh.r[455, :],handMesh.r[458:468, :],
                                  handMesh.r[429, :], handMesh.r[432:440, :], handMesh.r[442:445, :]])  # 24
            tip_v_dict[2] = mf_tip_v
            rf_tip_v = np.vstack([handMesh.r[543:551, :], handMesh.r[554:556, :], handMesh.r[559:562, :],
                                  handMesh.r[566, :], handMesh.r[569:579, :], handMesh.r[540, :]])  # 25
            tip_v_dict[3] = rf_tip_v
            lf_tip_v = np.vstack([handMesh.r[657:658, :], handMesh.r[660:668, :], handMesh.r[675:679, :],
                                  handMesh.r[670:673, :], handMesh.r[683, :], handMesh.r[686:696, :]])   # 27
            tip_v_dict[4] = lf_tip_v
            tip_v = np.vstack([th_tip_v, ff_tip_v, mf_tip_v, rf_tip_v, lf_tip_v])


        # project to 2D
        if split == 'train':
            handKps = project_3D_points(anno['camMat'], handJoints3D, is_OpenGL_coords=True)
        else:
            # Only root joint available in evaluation split
            handKps = project_3D_points(anno['camMat'], np.expand_dims(anno['handJoints3D'],0), is_OpenGL_coords=True)
        objKps = project_3D_points(anno['camMat'], objCornersTrans, is_OpenGL_coords=True)

        # Visualize
        if args['visType'] == 'open3d':
            # open3d visualization
            if not os.path.exists(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj')):
                raise Exception('3D object models not available in %s'%(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj')))

            # load object model
            objMesh = read_obj(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj'))
            mesh = trimesh.load(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj'))

            # apply current pose to the object model
            objMesh.v = np.matmul(objMesh.v, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']
            mesh.vertices = np.matmul(mesh.vertices, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']
            contact = np.zeros([5])
            for i in range(5):
                dis = trimesh.proximity.signed_distance(mesh, tip_v_dict[i])
                print(dis)
                if np.where(np.abs(dis) < 0.0005)[0].shape[0]:
                    contact[i] = 1
            print(contact)

            ## implicit interaction to emplicit interaction
            ## compared to build objec mesh, get contact info --> easier

            # show
            if split == 'train':
                open3dVisualize_tip([handMesh, objMesh], tip_v, tipJoints3D,  ['r', 'g'])
            else:
                open3dVisualize([objMesh], ['r', 'g'])

        # elif args['visType'] == 'matplotlib':

            # draw 2D projections of annotations on RGB image
            if split == 'train':
                imgAnno = showHandJoints(img, handKps[jointsMapManoToSimple])
            else:
                # show only projection of root joint in evaluation split
                imgAnno = showHandJoints(img, handKps)
                # show the hand bounding box
                imgAnno = show2DBoundingBox(imgAnno, anno['handBoundingBox'])
            imgAnno = showObjJoints(imgAnno, objKps, lineThickness=2)

            # create matplotlib window
            fig = plt.figure(figsize=(2, 2))
            figManager = plt.get_current_fig_manager()
           # figManager.resize(*figManager.window.maxsize())
            figManager.window.showMaximized()


            # show RGB image
            ax0 = fig.add_subplot(2, 2, 1)
            ax0.imshow(img[:, :, [2, 1, 0]])
            ax0.title.set_text('RGB Image')

            # show depth map
            ax1 = fig.add_subplot(2, 2, 2)
            ax1.imshow(depth)
            ax1.title.set_text('Depth Map')

            # show 3D hand mesh
            ax2 = fig.add_subplot(2, 2, 3, projection="3d")
            if split=='train':
                plot3dVisualize(ax2, handMesh, flip_x=False, isOpenGLCoords=True, c="r")
            ax2.title.set_text('Hand Mesh')

            # show 2D projections of annotations on RGB image
            ax3 = fig.add_subplot(2, 2, 4)
            ax3.imshow(imgAnno[:, :, [2, 1, 0]])
            ax3.title.set_text('3D Annotations projected to 2D')

            plt.show()
        else:
            raise Exception('Unknown visualization type')

        if runLoop:
            args['seq'] = random.choice(os.listdir(join(baseDir, split)))
            args['id'] = random.choice(os.listdir(join(baseDir, split, args['seq'], 'rgb'))).split('.')[0]
        else:
            break
