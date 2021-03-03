import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from __future__ import print_function
import argparse
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
parser.add_argument('--face-batch-size', type=int, default=64, metavar='')
parser.add_argument('--epochs', type=int, default=50, metavar='')
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


class Purifier(nn.Module):
	def __init__(self):
		super(Purifier, self).__init__()

		self.encoder = nn.Sequential(nn.Linear(530, 200),
									 nn.ReLU(True))

		self.decoder = nn.Sequential(nn.Linear(200, 530),
									 nn.ReLU(True))
								
	def forward(self, x):
		encode = self.encoder(x)
		decode = self.decoder(encode)
		return encode, decode

def train(purifier, classifier, inversion, device, data_loader,optimizier, epoch ):

	classifier.eval()
	inversion.eval()

	#batch_size = 128

	alpha = 1
	beta = 1
	#optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	#l2norm = nn.MSELoss()


	for batch_idx, (data, target) in enumerate(data_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()


		logit = classifier(data，release = False)
		out = purifier(logit)
		pred = F.softmax(out, dim=1)
		recon = inversion(pred)

		loss = F.mse_loss(logit,out)
				+alpha * F.nll_loss(pred, target)
				- beta * F.mse_loss(recon, data)

		loss.backward()
		optimizier.step()
	print("epoch=", epoch, loss.data.float())

def test(purifier, classifier, inversion, device, data_loader ):

	classifier.eval()
	inversion.eval()

	l2norm = nn.MSELoss()
	diff = 0
	recon_err = 0
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device), target.to(device)

			logit = classifier(data，release = False)
			out = purifier(logit)
			pred = F.softmax(out, dim=1)
			recon = inversion(pred)
			diff += F.mse_loss(logit, out, reduction='sum').item()
			recon_err += F.mse_loss(recon, data, reduction='sum').item()
			test_loss += F.nll_loss(pred, target, reduction='sum').item()

			label = output.max(1, keepdim=True)[1]
			correct += pred.eq(target.view_as(label)).sum().item()


	diff /= len(data_loader.dataset)
	recon_err /= len(data_loader.dataset)
	test_loss /= len(data_loader.dataset)
	correct /= len(data_loader.dataset)
	print('diff:', diff)
	print('recon_err:', recon_err)
	print('test_loss:', test_loss)
	print('accu:',correct)
	print('**********************')


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


	train_set = FaceScrub('./facescrub.npz', transform=transform, train=True)
	test_set = FaceScrub('./facescrub.npz', transform=transform, train=False)

	train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

	classifier = nn.DataParallel(Classifier(nc=args.nc, ndf=args.ndf, nz=args.nz)).to(device)
	inversion = nn.DataParallel(Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.truncation, c=args.c)).to(device)
	purifier = nn.DataParallel(Purifier()).to(device)
	model_path = 'model/model_dict.pth'
	inversion_path = 'model/inversion.pth'
	#inversion_path = 'out/inversion_def.pth'
	
	lr = 5e-4
	weight_decay = 1e-5

	optimizier = optim.Adam(purifier.parameters(), lr=lr, weight_decay=weight_decay)


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

	best_acc = 0
	best_epoch = 0
	for epoch in range(1, args.epochs + 1):
		train(purifier, classifier, inversion, device, train_loader,optimizier, epoch)
		#test(purifier, classifier, inversion, device, data_loader )
		#cl_acc = test(classifier, device, test_loader)
		'''
		if cl_acc > best_cl_acc:
			best_cl_acc = cl_acc
			best_cl_epoch = epoch
			state = {
				'epoch': epoch,
				'model': classifier.state_dict(),
				'optimizer': optimizer.state_dict(),
				'best_cl_acc': best_cl_acc,
			}
			torch.save(state, 'model/classifier.pth')
			torch.save(classifier.state_dict(), 'model/model_dict.pth')
		'''
		torch.save(purifier.state_dict(), 'model/purifier_dict.pth')

	
	#print("Best classifier: epoch {}, acc {:.4f}".format(best_cl_epoch, best_cl_acc))
	

if __name__ == '__main__':
	main()
