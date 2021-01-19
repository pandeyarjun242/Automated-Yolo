import numpy as np
import cv2
import time
import os
class Yolov3Video:
    def __init__(self,videos,weights,cfg,names,vn):
        self.videos = videos
        self.weights = weights
        self.cfg = cfg
        self.names = names
        self.vn = vn
    def run_model(self): 
        video = self.videos #input
        writer = None
        h, w = None, None
        with open(self.names) as f:
            labels = [line.strip() for line in f]
        network = cv2.dnn.readNetFromDarknet(self.cfg,self.weights)

        layers_names_all = network.getLayerNames()
        layers_names_output = \
            [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

        probability_minimum = 0.5

        threshold = 0.3

        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
        lps = []
        f = 0
        t = 0
        while True:
            # Capturing frame-by-frame
            ret, frame = video.read()

            # If the frame was not retrieved
            # e.g.: at the end of the video,
            # then we break the loop
            if not ret:
                break

            # Getting spatial dimensions of the frame
            # we do it only once from the very beginning
            # all other frames have the same dimension
            if w is None or h is None:
                # Slicing from tuple only first two elements
                h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)


            # Implementing forward pass with our blob and only through output layers
            # Calculating at the same time, needed time for forward pass
            network.setInput(blob)  # setting blob as input to the network
            start = time.time()
            output_from_network = network.forward(layers_names_output)
            end = time.time()

            # Increasing counters for frames and total time
            f += 1
            t += end - start

            # Showing spent time for single current frame
            print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))
            yield('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

            bounding_boxes = []
            confidences = []
            class_numbers = []

            # Going through all output layers after feed forward pass
            for result in output_from_network:
                # Going through all detections from current output layer
                for detected_objects in result:
                    scores = detected_objects[5:]
                    class_current = np.argmax(scores)
                    confidence_current = scores[class_current]
                    if confidence_current > probability_minimum:
                        box_current = detected_objects[0:4] * np.array([w, h, w, h])

                        # Now, from YOLO data format, we can get top left corner coordinates
                        # that are x_min and y_min
                        x_center, y_center, box_width, box_height = box_current
                        x_min = int(x_center - (box_width / 2))
                        y_min = int(y_center - (box_height / 2))

                        # Adding results into prepared lists
                        bounding_boxes.append([x_min, y_min,
                                               int(box_width), int(box_height)])
                        confidences.append(float(confidence_current))
                        class_numbers.append(class_current)
            results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                       probability_minimum, threshold)
            if len(results) > 0:
                # Going through indexes of results
                for i in results.flatten():
                    x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                    box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                    colour_box_current = colours[class_numbers[i]].tolist()
                    # Drawing bounding box on the original current frame
                    roi = frame[y_min:y_min+box_height+10,x_min:x_min+box_width+10]  
                    cv2.rectangle(frame, (x_min, y_min),
                                  (x_min + box_width, y_min + box_height),
                                  colour_box_current, 2)
                    text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                           confidences[i])
                    cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
                
            if writer is None:
            
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter('result'+self.vn, fourcc, 30,
                                         (frame.shape[1], frame.shape[0]), True)
            name = 'result'+self.vn
            writer.write(frame)
        print()
        print('Total number of frames', f)
        print('Total amount of time {:.5f} seconds'.format(t))
        print('FPS:', round((f / t), 1))

        video.release()
        writer.release()
        yield name