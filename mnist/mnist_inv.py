from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os, shutil
#from data import FaceScrub, CelebA
#from model import Classifier, Inversion
import torch.nn.functional as F
import torchvision.utils as vutils
from mnist import Net, Inversion

# Training settings
parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch-size', type=int, default=256, metavar='')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='')
parser.add_argument('--epochs', type=int, default=20, metavar='')
parser.add_argument('--lr', type=float, default=0.01, metavar='')
parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')
parser.add_argument('--c', type=float, default=50.)
parser.add_argument('--num_workers', type=int, default=1, metavar='')

def train(classifier, inversion, log_interval, device, data_loader, optimizer, epoch):
	classifier.eval()
	inversion.train()

	for batch_idx, (data, target) in enumerate(data_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		with torch.no_grad():
			prediction = classifier(data, log = False)
		#print(prediction)
		reconstruction = inversion(prediction)
		#print(reconstruction)
		#print(data)
		loss = F.mse_loss(reconstruction, data)
		loss.backward()
		optimizer.step()

		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format( epoch, batch_idx * len(data),
																  len(data_loader.dataset), loss.item()))
	

def test(classifier, inversion, device, data_loader, msg):
	classifier.eval()
	inversion.eval()
	mse_loss = 0
	plot = True
	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device), target.to(device)

			prediction = classifier(data, log = False)
			reconstruction = inversion(prediction)
			mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

			if plot:
				truth = data[0:32]
				inverse = reconstruction[0:32]
				out = torch.cat((inverse, truth))
				for i in range(4):
					out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
					out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
				vutils.save_image(out, 'out/recon_{}.png'.format(msg.replace(" ", "")), normalize=False)
				plot = False

	mse_loss /= len(data_loader.dataset) * 64 * 64
	print('\nTest inversion model on {} set: Average MSE loss: {:.6f}\n'.format(msg, mse_loss))
	return mse_loss

def main():
	args = parser.parse_args()
	print("================================")
	print(args)
	print("================================")
	os.makedirs('out', exist_ok=True)

	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

	torch.manual_seed(args.seed)

	transform = transforms.Compose([transforms.ToTensor()])
	#train_set = CelebA('./celeba_5w_255.npy', transform=transform)
	# Inversion attack on TRAIN data of facescrub classifier
	#test1_set = FaceScrub('./facescrub.npz', transform=transform, train=False)
	# Inversion attack on TEST data of facescrub classifier
	#test2_set = FaceScrub('./facescrub.npz', transform=transform, train=False)
	train_set = datasets.QMNIST('../data', train=True, download=True,
					   transform=transform)
	test_set = datasets.QMNIST('../data', train=False, download=True,
					   transform=transform)
	test2_set = datasets.MNIST('../data', train=False, download=True,
					   transform=transform)
	#train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
	#test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
	test2_loader = torch.utils.data.DataLoader(test2_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

	classifier = nn.DataParallel(Net()).to(device)
	inversion = nn.DataParallel(Inversion()).to(device)
	optimizer = optim.Adam(inversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True)

	# Load classifier
	path = 'model/mnist_cnn.pth'
	#path = 'model/mnist_mi.pth'
	#checkpoint = torch.load(path)

	path = 'model/mnist_cnn.pth'
	inv_path =  'model/mnist_inv.pth'
	#checkpoint = torch.load(path)
	try:
		checkpoint = torch.load(path)
		classifier.load_state_dict(checkpoint)
	except:
		print("=> load classifier checkpoint '{}' failed".format(path))
		return
	try:
		inv_checkpoint = torch.load(inv_path)
		inversion.load_state_dict(inv_checkpoint)
	except:
		print("=> load classifier checkpoint '{}' failed".format(inversion_path))
		return

	test(classifier, inversion, device, test_loader, 'qmnist')
	test(classifier, inversion, device, test2_loader, 'mnist')


	'''
	# Train inversion model
	best_recon_loss = 999
	for epoch in range(1, args.epochs + 1):
		train(classifier, inversion, args.log_interval, device, train_loader, optimizer, epoch)
		recon_loss = test(classifier, inversion, device, test1_loader, epoch, 'test1')
		#test(classifier, inversion, device, test2_loader, epoch, 'test2')

		#break
		if recon_loss < best_recon_loss:
			best_recon_loss = recon_loss
			state = {
				'epoch': epoch,
				'model': inversion.state_dict(),
				'optimizer': optimizer.state_dict(),
				'best_recon_loss': best_recon_loss
			}
			#torch.save(state, 'model/inversion.pth')
			torch.save(inversion.state_dict(), 'model/mnist_mi_inv.pth')
			shutil.copyfile('out/recon_test1_{}.png'.format(epoch), 'out/best_test_mi.png')
			#shutil.copyfile('out/recon_test2_{}.png'.format(epoch), 'out/best_test2.png')
	'''

if __name__ == '__main__':
	main()
