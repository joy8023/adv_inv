from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os, shutil
from data import FaceScrub, CelebA, FaceScrub_out, CelebA_out
from model import Classifier, Inversion
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np

parser = argparse.ArgumentParser(description='defense against model inversion')
parser.add_argument('--celeb-batch-size', type=int, default=128, metavar='')
parser.add_argument('--face-batch-size', type=int, default=128, metavar='')
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--nz', type=int, default=530)
parser.add_argument('--truncation', type=int, default=530)
parser.add_argument('--c', type=float, default=50.)
parser.add_argument('--num_workers', type=int, default=1, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--epsilon', type = float, default = 0.5, metavar = '')
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


def perturb(prediction, epsilon, grad, logit_original, logit = True):

	sign = grad.sign()
	logit_new = prediction + epsilon * sign 
	
	logit_diff = F.l1_loss(logit_new, logit_original)
	#print('*************logit_diff:',logit_diff.item())

	#calculate accuracy for perturbed images
	original_label = torch.max(logit_original, 1)[1].cpu().numpy()
	new_label = torch.max(logit_new, 1)[1].cpu().numpy()

	orig_label_onehot = F.one_hot(torch.tensor(original_label), 530)
	orig_label_onehot = torch.tensor(orig_label_onehot, dtype=torch.uint8)

	# to keep the max unchanged
	logit_new[orig_label_onehot] = torch.max(logit_new, 1)[0]*1.01

	accu = np.sum(original_label == new_label)/original_label.shape[0]

	output = F.softmax(logit_new, dim=1)
	#print(accu)
	#output_diff = F.mse_loss(output, F.softmax(prediction, dim=1))
	#print('************output_diff', output_diff.item())
	if logit == True:
		return logit_new
	else:
		return output

#main outer function
def add_noise(classifier, inversion, device, data_loader, epsilon, num_step):

	classifier.eval()
	inversion.eval()

	diff = 0
	recon_err = 0
	correct = 0
	l1max = 0
	plot = True


	for batch_idx, (data_, target_) in enumerate(data_loader):


		data = data_.to(device)
		data_ = data_.to(device)
		target_ = target_.to(device)
		print('############batch:',batch_idx)

		with torch.no_grad():
			logit = classifier(data, logit = True)

		for i in range(num_step):

			#print('========perturbation iteration:',i)
			#create new tensor for further perturbation
			logit = logit.clone().detach().to(device)
			logit.requires_grad = True

			reconstruction = inversion(F.softmax(logit, dim=1))
			loss =F.mse_loss(reconstruction, data)
			loss.backward()

			logit_grad = logit.grad.data

			if i == 0:
				original_logit = logit

			perturbation = perturb(logit, epsilon, logit_grad, original_logit)
			#perturbation= perturbation.to(device)
			pert_recon = inversion(F.softmax(perturbation, dim=1))

			logit = perturbation

			#for plot use
			'''
			if i == 0:
				truth = data[0:8]
				inverse = reconstruction[0:8]
				out = torch.cat((truth, inverse))
				#out = torch.cat((out, pert_recon[0:8]))

			
			out = out = torch.cat((out, pert_recon[0:8]))
			'''
			#data = pert_recon.clone().detach().to(device)


		# test defense use
		
		l1 = F.l1_loss(original_logit, perturbation).max().item()
		if l1>l1max:
			l1max = l1

		diff += F.l1_loss(original_logit, perturbation, reduction='sum').item()
		recon_err += F.mse_loss(pert_recon, data_, reduction='sum').item()
		label = perturbation.max(1, keepdim=True)[1]
		correct += label.eq(target_.view_as(label)).sum().item()

		if plot:
			truth = data_[0:32]
			inverse = pert_recon[0:32]
			out = torch.cat((inverse, truth))
			for i in range(4):
				out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
				out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
			vutils.save_image(out, 'out/recon_def_adv.png', normalize=False)
			plot = False
	'''
		#to save img and their result
		if batch_idx == 0:
			img = data_
			result = perturbation
		else:

			img = torch.cat((img,data_))
			result = torch.cat((result,perturbation))

	img = img.cpu().numpy()
	result = result.detach().cpu().numpy()
	print(img.shape)
	print(result.shape)

	#generating dataset
	np.savez("celeba_def.npz", images = img, out = result)
	
	'''
	diff /= len(data_loader.dataset)*530
	recon_err /= len(data_loader.dataset)*64*64
	correct /= len(data_loader.dataset)
	print('diff:', diff)
	print('recon_err:', recon_err)
	print('accu:',correct)
	print('l1max:',l1max)
	print('**********************')
	

	return

#main outer function
def add_noise_2inv(classifier, inversion, inversion2, device, data_loader, epsilon, num_step):

	classifier.eval()
	inversion.eval()
	inversion2.eval()

	diff = 0
	recon_err = 0
	correct = 0
	l1max = 0
	plot = True


	for batch_idx, (data_, target_) in enumerate(data_loader):


		data = data_.to(device)
		data_ = data_.to(device)
		target_ = target_.to(device)
		print('############batch:',batch_idx)

		with torch.no_grad():
			logit = classifier(data, logit = True)

		for i in range(num_step):

			#print('========perturbation iteration:',i)
			#create new tensor for further perturbation
			logit = logit.clone().detach().to(device)
			logit.requires_grad = True

			reconstruction = inversion(F.softmax(logit, dim=1))
			loss =F.mse_loss(reconstruction, data)
			loss.backward()

			logit_grad = logit.grad.data

			if i == 0:
				original_logit = logit

			perturbation = perturb(logit, epsilon, logit_grad, original_logit)
			pert_recon = inversion(F.softmax(perturbation, dim=1))

			logit = perturbation

		# test defense use
		pert_recon = inversion2(F.softmax(perturbation, dim=1))

		l1 = F.l1_loss(original_logit, perturbation).max().item()
		if l1>l1max:
			l1max = l1

		diff += F.l1_loss(original_logit, perturbation, reduction='sum').item()
		recon_err += F.mse_loss(pert_recon, data_, reduction='sum').item()
		label = perturbation.max(1, keepdim=True)[1]
		correct += label.eq(target_.view_as(label)).sum().item()

		if plot:
			truth = data_[0:32]
			inverse = pert_recon[0:32]
			out = torch.cat((inverse, truth))
			for i in range(4):
				out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
				out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
			vutils.save_image(out, 'out/recon_def_adv222.png', normalize=False)
			plot = False

	diff /= len(data_loader.dataset)*530
	recon_err /= len(data_loader.dataset)*64*64
	correct /= len(data_loader.dataset)
	print('diff:', diff)
	print('recon_err:', recon_err)
	print('accu:',correct)
	print('l1max:',l1max)
	print('**********************')
	

	return



def inv_test(classifier, inversion, device, data_loader, epoch = 100, msg = 'test'):

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
				vutils.save_image(out, 'out2/recon_{}_{}_vanilla.png'.format(msg.replace(" ", ""), epoch), normalize=False)
				plot = False

	mse_loss /= len(data_loader.dataset) * 64 * 64
	print('\nTest inversion model on {} set: Average MSE loss: {:.6f}\n'.format(msg, mse_loss))
	return mse_loss

#test with output directly
def inv_test2(inversion, device, data_loader, epoch = 100, msg = 'test'):

	#classifier.eval()
	inversion.eval()
	mse_loss = 0
	plot = True

	with torch.no_grad():
		for data, out in data_loader:
			data, out = data.to(device), out.to(device)

			#prediction = classifier(data, release=True)
			reconstruction = inversion(out)
			mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

			if plot:
				truth = data[0:32]
				inverse = reconstruction[0:32]
				out = torch.cat((inverse, truth))
				for i in range(4):
					out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
					out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
				vutils.save_image(out, 'out2/recon_{}_{}_def.png'.format(msg.replace(" ", ""), epoch), normalize=False)
				plot = False

	mse_loss /= len(data_loader.dataset) * 64 * 64
	print('\nTest inversion model on defensed {} set: Average MSE loss: {:.6f}\n'.format(msg, mse_loss))
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

	#celeb_set = CelebA('./celeba_5w_255.npy', transform=transform)
	#face_set = FaceScrub_out('./face_out.npz', transform=transform, train=False)

	celeb_loader = torch.utils.data.DataLoader(celeb_set, batch_size=args.celeb_batch_size, shuffle=True, **kwargs)
	face_loader = torch.utils.data.DataLoader(face_set, batch_size=args.face_batch_size, shuffle=False, **kwargs)

	classifier = nn.DataParallel(Classifier(nc=args.nc, ndf=args.ndf, nz=args.nz)).to(device)
	inversion = nn.DataParallel(Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.truncation, c=args.c)).to(device)
	inversion2 = nn.DataParallel(Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.truncation, c=args.c)).to(device)

	model_path = 'model/model_dict.pth'
	inversion_path = 'model/inv_model.pth'
	inversion2_path = 'out/inv_def.pth'

	try:
		model_checkpoint = torch.load(model_path)
		classifier.load_state_dict(model_checkpoint)

	except:
		print("=> load classifier checkpoint '{}' failed".format(model_path))
		return
	try:
		model_checkpoint = torch.load(inversion2_path)
		inversion2.load_state_dict(model_checkpoint)

	except:
		print("=> load classifier checkpoint '{}' failed".format(model_path))
		return

	try:
		inv_checkpoint = torch.load(inversion_path)
		inversion.load_state_dict(inv_checkpoint)
	except:
		print("=> load classifier checkpoint '{}' failed".format(inversion_path))
		return
	
	epsilon = args.epsilon
	num_step = args.num_step
	
	#add_noise(classifier, inversion, device, celeb_loader, epsilon, num_step)
	add_noise_2inv(classifier, inversion, inversion2, device, face_loader, epsilon, num_step)
	#inv_test2(inversion, device, face_loader)
	#inv_test(classifier, inversion, device, face_loader)

if __name__ == '__main__':
	main()
