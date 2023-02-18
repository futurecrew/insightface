import sys
import cv2
import os
import glob
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

if __name__ == '__main__':
    if len(sys.argv) > 1:
        instance_dir = sys.argv[1]
    else:
        instance_dir = ''
    if len(sys.argv) > 2:
        req_id = sys.argv[2]
    else:
        req_id = ''
    if len(sys.argv) > 3:
        detail = sys.argv[3]
    else:
        detail = False
        
    debug = False

    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(512, 512))
    
    gender_man = 0
    gender_woman = 0
    est_total = 0
    age_total = 0
    detail_result = []
    
    files = glob.glob('%s/*.png' % instance_dir)
    files2 = glob.glob('%s/*.jpg' % instance_dir)
    files.extend(files2)
    
    for f in files:
        #f = '/media/dj/play/project/test/images/aaa.jpg'
        if os.path.isfile(f):
            img = cv2.imread(f, cv2.IMREAD_COLOR)
        
            faces = app.get(img)
            if len(faces) == 0:
                continue
            
            rimg = app.draw_on(img, faces)
        
            est_total += 1
            max_area = 0
            max_index = 0

            for i, one_face in enumerate(faces):
                """
                print(type(one_face))
                print(one_face.keys())
                print(one_face['bbox'])
                print(type(one_face['bbox']))
                """
                f_x1 = one_face['bbox'][0]
                f_y1 = one_face['bbox'][1]
                f_x2 = one_face['bbox'][2]
                f_y2 = one_face['bbox'][3]
                f_w = f_x2 - f_x1
                f_h = f_y2 - f_y1
                
                if f_w * f_h > max_area:
                    max_area = f_w * f_h
                    max_index = i

            face_max = faces[max_index]
            
            if debug:
                print(face_max['gender'])
                print(face_max['landmark_2d_106'])
                #print(face_max['landmark_3d_68'])
                
                for x, y in face_max['landmark_2d_106']:
                #for x, y, z in face_max['landmark_3d_68']:
                    print('x: %s, y: %s' % (x, y))
                    cv2.circle(rimg, (int(x), int(y)), 1, (0, 0, 255), 2)
                #cv2.imwrite("./t1_output.jpg", rimg)                
                cv2.imshow('rimg', rimg)
                cv2.waitKey(0)
            
            if face_max['gender'] == 1:
                gender_man += 1
            else:
                gender_woman += 1
                
            age_total += face_max['age']
            
            detail_info = {}
            detail_info['path'] = f
            detail_info['gender'] = face_max['gender']
            detail_info['age'] = face_max['age']
            detail_result.append(detail_info)
    
        if gender_man > gender_woman * 2:
            gender = 'man'
        elif gender_woman > gender_man * 2:
            gender = 'woman'
        else:
            gender = 'person'
    
    if est_total > 0:
        age = str(int(age_total/est_total))
            
        #print('est_man: %s, est_woman: %s, age: %s' % (gender_man, gender_woman, age))
    
        if os.path.exists('log') == False:
            os.makedirs('log')
        
        if detail:
            with open('log/insightface_%s.txt' % req_id, 'w') as fp:
                for one_info in detail_result:
                    fp.write(one_info['path'])
                    fp.write('\n')
                    if one_info['gender'] == 1:
                        fp.write('man')
                    else:
                        fp.write('woman')
                    fp.write('\n')
                    fp.write(str(one_info['age']))
                    fp.write('\n')
        else:
            with open('log/insightface_%s.txt' % req_id, 'w') as fp:
                fp.write(instance_dir)
                fp.write('\n')
                fp.write(gender)
                fp.write('\n')
                fp.write(age)
                fp.write('\n')
    else:
        print('cannot find any image files in %s' % instance_dir)
