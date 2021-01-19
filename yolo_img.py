import numpy as np
import cv2
import os
import time
import argparse
class YoloImg:
	def __init__(self,img,weights,cfg,names,file):
		self.img = img
		self.weights = weights
		self.names = names
		self.cfg = cfg
		self.image_BGR = img
		self.file = file
	def run_model(self):
		h, w = self.image_BGR.shape[:2]  # Slicing from tuple only first two elements

		blob = cv2.dnn.blobFromImage(self.image_BGR, 1 / 255.0, (416, 416),
		                             swapRB=True, crop=False)

		with open(self.names) as f: ## this is an input
		    labels = [line.strip() for line in f]

		network = cv2.dnn.readNetFromDarknet(self.cfg,self.weights) ## inputs

		layers_names_all = network.getLayerNames()

		layers_names_output = \
		    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

		probability_minimum = 0.7


		threshold = 0.3

		colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

		network.setInput(blob)  # setting blob as input to the network
		start = time.time()
		output_from_network = network.forward(layers_names_output)
		end = time.time()

		# Showing spent time for forward pass
		print('Objects Detection took {:.5f} seconds'.format(end - start))

		bounding_boxes = []
		confidences = []
		class_numbers = []


		# Going through all output layers after feed forward pass
		for result in output_from_network:
		    # Going through all detections from current output layer
		    for detected_objects in result:
		        # Getting 80 classes' probabilities for current detected object
		        scores = detected_objects[5:]
		        # Getting index of the class with the maximum value of probability
		        class_current = np.argmax(scores)
		        # Getting value of probability for defined class
		        confidence_current = scores[class_current]

		        if confidence_current > probability_minimum:
		            box_current = detected_objects[0:4] * np.array([w, h, w, h])
		            x_center, y_center, box_width, box_height = box_current
		            x_min = int(x_center - (box_width / 2))
		            y_min = int(y_center - (box_height / 2))
		                # Adding results into prepared lists
		            print(range(len(class_numbers)))
		            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
		            confidences.append(float(confidence_current))
		            class_numbers.append(class_current)

		results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
		                           probability_minimum, threshold)

		counter = 1
		objs = []
		objc = []
		arr = []
		# Checking if there is at least one detected object after non-maximum suppression
		if len(results) > 0:
		    plates = []
		    # Going through indexes of results
		    for i in results.flatten():
		        # Showing labels of the detected objects
		        print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))

		        counter += 1
		        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
		        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

		        # Preparing colour for current bounding box
		        # and converting from numpy array to list
		        colour_box_current = colours[class_numbers[i]].tolist()
		        # Drawing bounding box on the original image
		        roi = self.image_BGR[y_min:y_min+box_height+10,x_min:x_min+box_width+10]

		        plates.append(roi)
		        cv2.rectangle(self.image_BGR, (x_min, y_min),
		                      (x_min + box_width, y_min + box_height),
		                      colour_box_current, 2)

		        # Preparing text with label and confidence for current bounding box
		        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
		                                               confidences[i])
		        cv2.putText(self.image_BGR, text_box_current, (x_min, y_min - 5),
		                    cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

		for i in range(len(objs)):
		    cv2.imwrite('image')
		print()
		print('Total objects been detected:', len(bounding_boxes))
		print('Number of objects left after non-maximum suppression:', counter - 1)
		
		print('Yayy just made some detections')
		n = self.file[:-4]
		cv2.imwrite(n+'result.jpg',self.image_BGR)
		return self.image_BGR