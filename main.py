import os
import cv2
import time
import torch
import argparse
import numpy as np

from rtmp import rtmpPipe

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('--camera', default=source,  # required=True,  # default=2,
                        help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=384,
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                        help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                        help='Backbone model for SPPE FastPose model.')
    par.add_argument('--save_out', type=str, default='',
                        help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on cpu or cuda.')
    par.add_argument('--nms', type=float, default=0.01,
                        help='Non-Maximum Suppression overlap threshold.')
    par.add_argument('--conf_thres', type=float, default=0.99,
                        help='Minimum Confidence threshold of predicted bboxs to cut off.')
    args = par.parse_args()

    device = args.device
    nms = args.nms
    conf_thres = args.conf_thres

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device, nms=nms, conf_thres=conf_thres)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

    # rtmpUrl="rtmp://192.168.100.240:1990/live/livestream31"
    rtmpUrl="rtmp://192.168.100.240/live/livestream31"
    pipe=rtmpPipe()

    streamss = cv2.VideoCapture(cam_source)
    streamss.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fpsss = streamss.get(cv2.CAP_PROP_FPS)
    # fpsss = 25.0
    frame_sizess = (int(streamss.get(cv2.CAP_PROP_FRAME_WIDTH)), int(streamss.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    pipe.createPipe(768, 768, fpsss, rtmpUrl)
    # pipe.createPipe(frame_sizess[0], frame_sizess[1], fpsss, rtmpUrl)

    fps_time = 0
    f = 0
    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        image = frame.copy()

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        if detected is not None:
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = '{}'.format(action_name)
                if action_name == 'Fall Down':
                    clr = (255, 0, 0)
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)

            # VISUALIZE
            if track.time_since_update == 0:
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), clr, 1)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]
        fps_time = time.time()

        if outvid:
            writer.write(frame)

        cv2.imshow('frame', frame)
        pipe.send(frame)
        # print(frame.shape)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear resource. 
    cam.stop()
    if outvid:
        writer.release()
    cv2.destroyAllWindows()
