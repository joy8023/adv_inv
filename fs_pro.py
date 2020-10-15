import numpy as np
import os
import cv2


#original face images are organized as follows:
#.
#└──faces
#	└──names
#		└──images.jpeg

def getIMG(path):

	if not os.path.exists(path):
		print("no such directory")
		return

	images = []
	labels = []
	names = []

	C = 0

	with os.scandir(path) as dirs:
		for actor in dirs:
			if actor.is_dir():

				img_list = os.listdir(actor)
				for item in img_list:
					#print(item)
					if os.path.splitext(item)[1] == '.jpeg':
						img = cv2.imread(os.path.join(actor,item), cv2.IMREAD_GRAYSCALE)
						img = cv2.resize(img,(64,64))
						images.append(img)
						labels.append(C)

				names.append(actor.name)
				C += 1
				print(actor.name,C)


		images = np.array(images)
		labels = np.array(labels)
		print(images.shape)
		print(labels.shape)
		print(labels)
		print(len(names))
		#convert labels to one hot vector
		#one_hot = np.eye(len(names))[labels]

		np.savez("facescrub.npz", images = images, labels = labels, names = names)

getIMG("./faces")

#input = np.load("./facescrub.npz")
#print(input)
#data = input['images']
#labels = input['labels']
#names = input['names']
#print(labels)
#one = np.where(labels == 2)
#print(one)
#one_hot = np.eye(len(names))[labels]

#np.savez("facescrub.npz", images = data, labels = one_hot, names = names)




