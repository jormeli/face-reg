import os
import cv2
import sys
import numpy as np
from skimage import io
import dlib
import openface
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

train = False
if "--train" in sys.argv and len(sys.argv) - 1 > sys.argv.index("--train"):
    train = True
    train_id = sys.argv.index("--train") + 1

X = []
y = []
for file in os.listdir('./facedata/'):
    face_id = int(os.path.basename(os.path.splitext(file)[0]))
    for v in np.loadtxt('./facedata/'+file, delimiter=','):
        X.append(v)
        y.append(face_id)

X = np.array(X)
y = np.array(y)

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')

knn.fit(X, y)

video_capture = cv2.VideoCapture(0)
predictor_model = "shape_predictor_68_face_landmarks.dat"

face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)

win = dlib.image_window()

net = openface.TorchNeuralNet('nn4.small2.v1.t7', imgDim=96, cuda=False)

def detect_face(mat):
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    face_rect = face_aligner.getLargestFaceBoundingBox(mat)
    if face_rect is None:
        return None

    aligned_face = face_aligner.align(96, mat, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    return aligned_face

def embed_face(face):
    return net.forward(face)

def train(name):
    vecs = []
    while True:
        ret,frame= video_capture.read()
        win.clear_overlay()

        face = detect_face(frame)
        if face is None:
            continue

        win.set_image(face)
        vec = np.array(embed_face(face))
        if len(vecs) == 0:
            vecs.append(vec)
            continue

        avg = np.mean(vecs, axis=0)

        corr = np.corrcoef(avg, vec)[0][1]

        if corr < 0.9:
            print(corr)
            vecs.append(vec)

        if len(vecs) == 10:
            f = open('facedata/'+str(name)+'.txt', 'w')
            for v in vecs:
                f.write(','.join([str(i) for i in v]) + '\n')
            f.close()
            break

def predict():
    while True:
        ret,frame= video_capture.read()
        win.clear_overlay()

        face = detect_face(frame)
        if face is None:
            continue

        win.set_image(face)
        vec = embed_face(face).reshape(1,-1)
        neighbors = knn.kneighbors(vec, return_distance=False)
        dist = X[neighbors].dot(vec.T)
        kissa = np.nonzero(dist.flatten() > 0.8)
        classes = y[neighbors.flatten()[kissa]]
        if len(classes) < 1: 
            print("-1") 
        else:
            bincounts = np.bincount(classes)
            print(np.argmax(bincounts))

if train:
    train(train_id)
else:
    predict()

dlib.hit_enter_to_continue()
