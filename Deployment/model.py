import cv2
import numpy as np
import os
import easyocr
import re

# settings 
input_width = 640
input_height = 640

# Load Model
net = cv2.dnn.readNetFromONNX('model/license_plate_model.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def get_detections(img):
    # Convert Image to YOLO Format
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc,max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # Get Prediction From YOLO Model

    blob = cv2.dnn.blobFromImage(input_image,1/255,(input_width, input_height), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections

def non_maximum_suppression(input_image, detections):
    # Filter Detections Based on Confidence and Probability Score
    # centerx, center y, w, h, confidence, probability
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/input_width
    y_factor = image_h/input_height

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]

        if confidence > 0.4: # Confidence of detecting license plate
            class_score = row[5] # Probability score of license plate
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy - 0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # Clean
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # Non Maximum Suppression
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)
    
    return boxes_np, confidences_np, index

def blur(x,y,h,w, img):
    blurred_image = img
    blurred_image[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w] ,(700,700))
    return blurred_image

def draw_bb(image, boxes_np, confidences_np, index, filename):
    image_blur = image.copy()
    # Draw Bounding Box
    for idx in index:
        x, y, w, h = boxes_np[idx]
        bb_conf = confidences_np[idx]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        license_text, license_text_result, bbox = extract_text(image, boxes_np[idx])
    
        cv2.rectangle(image, (x,y-40), (x+w+85,y), (254,0,0),-1) # Box Containing Confidence Score     
        cv2.rectangle(image, (x,y+h), (x+w+85, y+h+30), (0,0,0), -1) # Box Containing OCR Result
        
        cv2.putText(image, conf_text, (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1) # Confidence Score Text
        cv2.putText(image, license_text, (x, y+h+25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 1) # OCR Result Text
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,80,255), 2) # Bounding Box

        cv2.imwrite('static/predict/{}'.format(filename), image) # Save Image Prediction Result
        roi = image[y:y+h, x:x+w]
        cv2.imwrite('static/roi/{}'.format(filename), roi) # Save ROI Result
        
        image_blur = blur(x,y,h,w,image_blur)
        cv2.imwrite('static/blurred_image/{}'.format(filename), image_blur) # Save image with blurred ROI
        roi_blur = image_blur[y:y+h, x:x+w]
        cv2.imwrite('static/blurred_roi/{}'.format(filename), roi_blur) # Save the blurred ROI

    return license_text, license_text_result

def extract_text(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]

    if 0 in roi.shape:
        return ''
    else: 
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(roi)
        
        text = ''
        for i in ocr_result:
            temp = re.sub(r'\s+','',i[1])
            text = text + " "+ temp

        text = ' '.join(text.split()).upper()
        
        text_result = re.match('\S+\s\S+\s\S+', text)
        print(text_result)
        if text_result:
            text_result = re.sub(r'[^\w\s]', '', text_result.group())
            return text,text_result, bbox
        else:
            return text,text_result, bbox

# Predictions
def yolo_predictions(image_location, filename):
    image = cv2.imread(image_location)
    ## Step-1: Detections
    input_image, detections = get_detections(image)
    ## Step-2: NMS
    boxes_np, confidences_np, index = non_maximum_suppression(input_image, detections)
    ## Step-3: Draw Bounding Box
    ocr_result, ocr_result_processed, = draw_bb(image, boxes_np, confidences_np, index, filename)
    
    return ocr_result, ocr_result_processed