from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os, shutil
from data import FaceScrub, CelebA
from model import Classifier, Inversion
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np

parser = argparse.ArgumentParser(description='defense against model inversion')
parser.add_argument('--celeb-batch-size', type=int, default=256, metavar='')
parser.add_argument('--face-batch-size', type=int, default=64, metavar='')
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--nz', type=int, default=530)
parser.add_argument('--truncation', type=int, default=530)
parser.add_argument('--c', type=float, default=50.)
parser.add_argument('--num_workers', type=int, default=1, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--epsilon', type = float, default = 10e-5, metavar = '')
parser.add_argument('--num_step', type = int, default = 10, metavar = '')


def predict(classifier, device, data_loader):

	classifier.eval()
	
	with torch.no_grad():
		for data, target in date_loader:
			data, target = data.to(device), target.to(device)
			output = classifier(data)

	return output

#the function to keep the out similar to the original one
#at least the classification is the same
def restore(new_logit, new_label, original_logit, original_label, amplifier = 1.1):

	if original_logit[original_label] > new_logit[new_label]:
		#print()
		return new_logit
	else:
		new_logit[original_label] = new_logit[new_label]*amplifier
		return new_logit


def perturb(prediction, epsilon, grad, logit_original):

	sign = grad.sign()
	logit_new = prediction + epsilon * sign 
	
	logit_diff = F.mse_loss(logit_new, logit_original)
	#print('*************logit_diff:',logit_diff.item())

	#calculate accuracy for perturbed images
	original_label = torch.max(logit_original, 1)[1].cpu().numpy()
	new_label = torch.max(logit_new, 1)[1].cpu().numpy()

	'''
	#to use the restore function to replace the max one by one
	for i in range(64):
		logit_new[i] = restore(logit_new[i],new_label[i], logit_original[i], original_label[i])

	'''

	#print(logit_original.size())
	orig_label_onehot = F.one_hot(torch.tensor(original_label), 530)
	#print(orig_label_onehot)
	orig_label_onehot = torch.tensor(orig_label_onehot, dtype=torch.uint8)
	#print(orig_label_onehot.size())
	#new_label_onehot = F.one_hot(torch.tensor(new_label), 530)

	# to keep the max unchanged
	logit_new[orig_label_onehot] = torch.max(logit_new, 1)[0]*1.1


	new_label = torch.max(logit_new, 1)[1].cpu().numpy()

	accu = np.sum(original_label == new_label)/original_label.shape[0]

	#print('************accu:',accu)

	'''
	print(torch.max(logit_original,1)[0].data)
	print(logit_new[0][original_label[0]].item())
	print(logit_new[0][new_label[0]].item())
	print(torch.max(logit_new,1)[0].data)
	'''

	output = F.softmax(logit_new, dim=1)
	output_diff = F.mse_loss(output, F.softmax(prediction, dim=1))
	#print('************output_diff', output_diff.item())

	return output

#main outer function
def add_noise(classifier, inversion, device, data_loader, epsilon, num_step):

	classifier.eval()
	inversion.eval()

	#ori_img = []
	#noise_logit = []

	for batch_idx, (data_, target_) in enumerate(data_loader):

		data, target = data_.to(device), target_.to(device)
		print('############batch:',batch_idx)

		for i in range(num_step):

			#print('========perturbation iteration:',i)

			#prediction = classifier(data, release = True)
			with torch.no_grad():
				logit = classifier(data, release = False)
		
			#create new tensor for further perturbation
			logit = torch.tensor(logit).to(device)
			logit.requires_grad = True

			reconstruction = inversion(F.softmax(logit, dim=1))
			loss =F.mse_loss(reconstruction, data)
			loss.backward()

			logit_grad = logit.grad.data

			if i == 0:
				original_logit = logit

			perturbation = perturb(logit, epsilon, logit_grad, original_logit)
			perturbation= perturbation.to(device)
			pert_recon = inversion(perturbation)

			#for plot use
			'''
			if i == 0:
				truth = data[0:8]
				inverse = reconstruction[0:8]
				out = torch.cat((truth, inverse))
				#out = torch.cat((out, pert_recon[0:8]))

			
			out = out = torch.cat((out, pert_recon[0:8]))
			'''
			data = torch.tensor(pert_recon).to(device)
		
		#to save img and their result
		if batch_idx == 0:
			img = data
			result = perturbation

		img = torch.cat((img,data))
		result = torch.cat((result,perturbation))


		#only do the first batch
		#break

	img = img.cpu().numpy()
	result = result.detach().cpu().numpy()
	print(img.shape)
	print(result.shape)

	np.savez("celeba_5w_out.npz", images = img, out = result)
	
	#out = torch.cat((out, torch.tensor(after_noise)))
	#vutils.save_image(out, 'out1/test_epsilon_{}.png'.format(epsilon), normalize=False)

	return

#old version
def defense(classifier, inversion, device, data_loader, epsilon):

	classifier.eval()
	inversion.eval()
	#epsilon = 1

	for batch_idx, (data, target) in enumerate(data_loader):

		data, target = data.to(device), target.to(device)
		prediction = classifier(data, release = True)

		logit = classifier(data, release = False)
		
		#create new tensor for further perturbation
		logit = torch.tensor(logit).to(device)
		logit.requires_grad = True

		reconstruction = inversion(F.softmax(logit, dim=1))
		loss =F.mse_loss(reconstruction, data)
		loss.backward()

		#print('grad:',logit.grad.data)
		logit_grad = logit.grad.data

		#print('grad size:', logit_grad.size())
		perturbation = perturb(logit, epsilon, logit_grad)
		perturbation= perturbation.to(device)
		pert_recon = inversion(perturbation)
		
		print(reconstruction[0].data)
		print(pert_recon[0].data)

		plot = False
		if plot:
			truth = data[0:32]
			inverse = reconstruction[0:32]
			defense = pert_recon[0:32]
			out = torch.cat((truth, inverse))
			out = torch.cat((out, defense))
			for i in range(4):
					out[i * 24 :i * 24 + 8] = truth[i * 8:i * 8 + 8]
					out[i * 24 + 8:i * 24 + 16] = inverse[i * 8:i * 8 + 8]
					out[i * 24 + 16:i * 24 + 24] = defense[i * 8:i * 8 + 8]
			vutils.save_image(out, 'out1/test_epsilon_{}.png'.format(epsilon), normalize=False)
			plot = False


		return


def inv_test(classifier, inversion, device, data_loader, epoch, msg = 'test'):

	classifier.eval()
	inversion.eval()
	mse_loss = 0
	plot = True

	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device), target.to(device)

			prediction = classifier(data, release=True)
			reconstruction = inversion(prediction)
			mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

			if plot:
				truth = data[0:32]
				inverse = reconstruction[0:32]
				out = torch.cat((inverse, truth))
				for i in range(4):
					out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
					out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
				vutils.save_image(out, 'out/recon_{}_{}.png'.format(msg.replace(" ", ""), epoch), normalize=False)
				plot = False

	mse_loss /= len(data_loader.dataset) * 64 * 64
	print('\nTest inversion model on {} set: Average MSE loss: {:.6f}\n'.format(msg, mse_loss))
	return mse_loss



#generate CelebA dataset with defensive logit
def celeb_logit(path):

	if not os.path.exists(path):
		print("no such directory")
		return

	images = []
	C = 0

	with os.scandir(path) as files:
		
		for file in files:
			if os.path.splitext(file)[1] == '.jpg' and C < 50000:
				img = cv2.imread(os.path.join(file), cv2.IMREAD_GRAYSCALE)
				#print(img.shape)
				img = cv2.resize(img[70:178,35:143],(64,64))
				#cv2.imshow("img",img)
				#cv2.waitKey(0)
				images.append(img)
				C += 1
	
	images = np.array(images)
	#images = images / 255.0

	print(images.shape)
	#print(images.max())
	np.save("celeba_5w_255.npy", images)


def main():

	args = parser.parse_args()
	print("================================")
	print(args)
	print("================================")

	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
	
	torch.manual_seed(args.seed)

	transform = transforms.Compose([transforms.ToTensor()])


	celeb_set = CelebA('./celeba_5w_255.npy', transform=transform)
	# Inversion attack on TRAIN data of facescrub classifier
	face_set = FaceScrub('./facescrub.npz', transform=transform, train=False)

	celeb_loader = torch.utils.data.DataLoader(celeb_set, batch_size=args.celeb_batch_size, shuffle=True, **kwargs)
	face_loader = torch.utils.data.DataLoader(face_set, batch_size=args.face_batch_size, shuffle=False, **kwargs)

	classifier = nn.DataParallel(Classifier(nc=args.nc, ndf=args.ndf, nz=args.nz)).to(device)
	inversion = nn.DataParallel(Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.truncation, c=args.c)).to(device)

	model_path = 'out/model_dict.pth'
	inversion_path = 'out/inversion.pth'

	try:
		model_checkpoint = torch.load(model_path)
		#print(model_checkpoint)
		classifier.load_state_dict(model_checkpoint)

	except:
		print("=> load classifier checkpoint '{}' failed".format(model_path))
		return

	try:
		inv_checkpoint = torch.load(inversion_path)
		#print(inv_checkpoint)
		inversion.load_state_dict(inv_checkpoint['model'])
	except:
		print("=> load classifier checkpoint '{}' failed".format(inversion_path))
		return
	
	epsilon = args.epsilon
	num_step = args.num_step

	'''
	for i in range(10):
		defense(classifier, inversion, device, celeb_loader,epsilon)
		epsilon *= 10
	'''
	#defense(classifier, inversion, device, celeb_loader,epsilon)
	add_noise(classifier, inversion, device, celeb_loader, epsilon, num_step)


if __name__ == '__main__':
	main()
