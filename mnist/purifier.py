from __future__ import print_function
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os, shutil
import torch.nn.functional as F
import torchvision.utils as vutils
from mnist import Net, Inversion
import numpy as np


parser = argparse.ArgumentParser(description='defense against model inversion')
parser.add_argument('--batch-size', type=int, default=256, metavar='')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='')
parser.add_argument('--epochs', type=int, default=20, metavar='')
parser.add_argument('--num_workers', type=int, default=1, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')


class Purifier(nn.Module):
	def __init__(self):
		super(Purifier, self).__init__()

		self.encoder = nn.Sequential(nn.Linear(10, 7),
									 nn.ReLU(True),
									 nn.Linear(7,4),
									 nn.BatchNorm1d(num_features=4))

		self.decoder = nn.Sequential(nn.Linear(4,7),
									#nn.BatchNorm1d(num_features=530),
									nn.ReLU(True),
									nn.Linear(7,10),
									nn.BatchNorm1d(num_features=10))
								
	def forward(self, x):
		encode = self.encoder(x)
		decode = self.decoder(encode)
		return encode, decode

def old_test(model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.l1_loss(output, target, reduction='sum').item()  # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)*10

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

def train(purifier, classifier, inversion, device, data_loader,optimizier, epoch ):

	classifier.eval()
	inversion.eval()

	a = 1
	b = 1
	c = 1

	for batch_idx, (data, target) in enumerate(data_loader):
		data, target = data.to(device), target.to(device)
		optimizier.zero_grad()


		logit = classifier(data, logit = True)
		_, out = purifier(logit)
		pred = F.softmax(out, dim=1)
		recon = inversion(pred)

		diff = F.mse_loss(logit, out)
		recon_err = F.mse_loss(recon, data)
		test_loss = F.nll_loss(F.log_softmax(out, dim = 1), target)

		#loss = (F.mse_loss(logit,out)
		#   + alpha * F.nll_loss(pred, target)
		#   - beta * F.mse_loss(recon, data))
		loss = a * diff - b * recon_err + c * test_loss 
		#loss = diff
		loss.backward()
		optimizier.step()

		label1 = logit.argmax(dim=1, keepdim=True)
		correct1 = label1.eq(target.view_as(label1)).sum().item()

		label = out.argmax(dim=1, keepdim=True)
		correct = label.eq(target.view_as(label)).sum().item()


		if batch_idx % 10 == 0:
			print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format( epoch, batch_idx * len(data), len(data_loader.dataset), loss.item()))
			print('diff:{:.6f}\trecon err:{:.6f}\ttest loss:{:.6f}'.format(diff.item(),recon_err.item(),test_loss.item()))
			print(correct1/correct)
	print("epoch=", epoch, loss.data.float())

def test(purifier, classifier, inversion, device, data_loader,msg ):

	classifier.eval()
	inversion.eval()

	l2norm = nn.MSELoss()
	diff = 0
	recon_err = 0
	test_loss = 0
	correct = 0
	plot = True
	l1max = 0
	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device), target.to(device)

			logit = classifier(data, logit = True)
			_, out = purifier(logit)
			pred = F.softmax(out, dim=1)
			recon = inversion(pred)

			l1 = F.l1_loss(logit, out).max().item()
			if l1>l1max:
				l1max = l1


			diff += F.l1_loss(logit, out, reduction='sum').item()
			recon_err += F.mse_loss(recon, data, reduction='sum').item()
			test_loss += F.nll_loss(pred, target, reduction='sum').item()

			label = out.max(1, keepdim=True)[1]
			correct += label.eq(target.view_as(label)).sum().item()

			if plot:
				truth = data[0:32]
				inverse = recon[0:32]
				out = torch.cat((inverse, truth))
				for i in range(4):
					out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
					out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
				vutils.save_image(out, 'out/recon_purifier_{}.png'.format(msg.replace(" ", "")), normalize=False)
				plot = False


	diff /= len(data_loader.dataset)*10
	recon_err /= len(data_loader.dataset)*28*28
	test_loss /= len(data_loader.dataset)
	correct /= len(data_loader.dataset)
	print('diff:', diff)
	print('recon_err:', recon_err)
	print('test_loss:', test_loss)
	print('accu:',correct)
	print('l1max:',l1max)
	print('**********************')


def mi_test(classifier, classifier_mi, inversion, device, data_loader, msg ):

	classifier.eval()
	inversion.eval()

	l2norm = nn.MSELoss()
	diff = 0
	recon_err = 0
	test_loss = 0
	correct = 0
	plot = True
	l1max = 0
	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device), target.to(device)

			logit = classifier(data, logit = True)
			out = classifier_mi(data,logit = True)

			#_, out = purifier(logit)
			pred = F.softmax(out, dim=1)
			recon = inversion(pred)

			l1 = F.l1_loss(logit, out).max().item()
			if l1>l1max:
				l1max = l1


			diff += F.l1_loss(logit, out, reduction='sum').item()
			recon_err += F.mse_loss(recon, data, reduction='sum').item()
			#test_loss += F.nll_loss(pred, target, reduction='sum').item()

			label = out.max(1, keepdim=True)[1]
			correct += label.eq(target.view_as(label)).sum().item()

			if plot:
				truth = data[0:32]
				inverse = recon[0:32]
				out = torch.cat((inverse, truth))
				for i in range(4):
					out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
					out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
				vutils.save_image(out, 'out/recon_mi_{}.png'.format(msg.replace(" ", "")), normalize=False)
				plot = False


	diff /= len(data_loader.dataset)*10
	recon_err /= len(data_loader.dataset)*28*28
	#test_loss /= len(data_loader.dataset)
	correct /= len(data_loader.dataset)
	print('diff:', diff)
	print('recon_err:', recon_err)
	#print('test_loss:', test_loss)
	print('accu:',correct)
	print('l1max:',l1max)
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

	train_set = datasets.QMNIST('../data', train=True, download=True,
					transform=transform)
	test_set = datasets.QMNIST('../data', train=False, download=True,
					transform=transform)
	test2_set = datasets.MNIST('../data', train=False, download=True,
					transform=transform)

	train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
	test2_loader = torch.utils.data.DataLoader(test2_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
	#train_set = FaceScrub('./facescrub.npz', transform=transform, train=True)
	#test_set = FaceScrub('./facescrub.npz', transform=transform, train=False)

	#train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
	#test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
	classifier = nn.DataParallel(Net()).to(device)
	classifier_mi = nn.DataParallel(Net()).to(device)
	inversion = nn.DataParallel(Inversion()).to(device)
	#classifier = nn.DataParallel(Ne(nc=args.nc, ndf=args.ndf, nz=args.nz)).to(device)
	#inversion = nn.DataParallel(Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.truncation, c=args.c)).to(device)
	purifier = nn.DataParallel(Purifier()).to(device)

	model_path = 'model/mnist_cnn.pth'
	inversion_path = 'model/mnist_inv.pth'	
	optimizier = optim.Adam(purifier.parameters(), lr=0.02, betas=(0.5, 0.999), amsgrad=True)
	
	try:
		model_checkpoint = torch.load(model_path)
		classifier.load_state_dict(model_checkpoint)

	except:
		print("=> load classifier checkpoint '{}' failed".format(model_path))
		return
	
	try:
		inv_checkpoint = torch.load(inversion_path)
		inversion.load_state_dict(inv_checkpoint)
	except:
		print("=> load classifier checkpoint '{}' failed".format(inversion_path))
		return

	best_acc = 0
	best_epoch = 0
	#purifier.load_state_dict(torch.load('model/purifier.pth'))

	#mi model test
	mi_path = 'model/mnist_mi.pth'
	try:
		checkpoint = torch.load(mi_path)
		classifier.load_state_dict(checkpoint)
	except:
		print("=> load classifier checkpoint '{}' failed".format(path))
		return


	#old_test(classifier, device, train_loader)
	mi_test(classifier, classifier_mi, inversion, device, test_loader,'qmnist')
	old_test(classifier_mi, device,test_loader)
	mi_test(classifier, classifier_mi, inversion, device, test2_loader,'mnist')

	return

	test(purifier, classifier, inversion, device, test_loader,'qmnist')
	test(purifier, classifier, inversion, device, test2_loader,'mnist')

	return

	for epoch in range(1, args.epochs + 1):
		train(purifier, classifier, inversion, device, train_loader,optimizier, epoch)
		test(purifier, classifier, inversion, device, test_loader )
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
		torch.save(purifier.state_dict(), 'model/purifier.pth')

	
	#print("Best classifier: epoch {}, acc {:.4f}".format(best_cl_epoch, best_cl_acc))
	

if __name__ == '__main__':
	main()
