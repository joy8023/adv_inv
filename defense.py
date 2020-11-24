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
parser.add_argument('--celeb-batch-size', type=int, default=64, metavar='')
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
parser.add_argument('--epsilon', type = int, default = 1, metavar = '')
parser.add_argument('--num_step', type = int, default = 10, metavar = '')




def predict(classifier, device, data_loader):

	classifier.eval()
	
	with torch.no_grad():
		for data, target in date_loader:
			data, target = data.to(device), target.to(device)
			output = classifier(data)

	return output

def perturb(prediction, epsilon, grad, original_logit):

	sign = grad.sign()
	logit_new = prediction + epsilon * sign 
	
	logit_diff = F.mse_loss(logit_new, original_logit)
	print('*************logit_diff:',logit_diff.item())
	#print(prediction[0].data)
	#print(output[0].data)
	original_label = torch.max(original_logit, 1)[1].cpu().numpy()
	new_label = torch.max(logit_new, 1)[1].cpu().numpy()
	#print(original_label)
	print(new_label.shape)

	accu = np.sum(original_label == new_label)/original_label.shape[0]
	print('************accu:',accu)


	output = F.softmax(logit_new, dim=1)
	output_diff = F.mse_loss(output, F.softmax(prediction, dim=1))
	print('************output_diff', output_diff.item())



	return output


def add_noise(classifier, inversion, device, data_loader, epsilon, num_step):

	classifier.eval()
	inversion.eval()
	#epsilon = 1

	after_noise = []

	for batch_idx, (data, target) in enumerate(data_loader):

		data, target = data.to(device), target.to(device)

		for i in range(num_step):

			print('********************perturbation iteration:',i)

			#prediction = classifier(data, release = True)
			logit = classifier(data, release = False)
		
			#create new tensor for further perturbation
			logit = torch.tensor(logit).to(device)
			logit.requires_grad = True

			reconstruction = inversion(F.softmax(logit, dim=1))
			loss =F.mse_loss(reconstruction, data)
			loss.backward()

			#print('grad:',logit.grad.data)
			logit_grad = logit.grad.data

			if i == 0:
				original_logit = logit

			#print('grad size:', logit_grad.size())
			perturbation = perturb(logit, epsilon, logit_grad, original_logit)
			perturbation= perturbation.to(device)
			pert_recon = inversion(perturbation)

			if i == 0:
				truth = data[0:8]
				inverse = reconstruction[0:8]
				out = torch.cat((truth, inverse))


			after_noise.append(pert_recon[0:8])
		break

	out = torch.cat((out, torch.tensor(after_noise)))
	vutils.save_image(out, 'out1/test_epsilon_{}.png'.format(epsilon), normalize=False)


		
			#print(reconstruction[0].data)
			#print(pert_recon[0].data)
	'''
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
	'''

	return

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
	add_noise(classifier, inversion, device, celeb_loader,epsilon, num_step)


if __name__ == '__main__':
	main()
