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


def predict(classifier, device, data_loader):

    classifier.eval()
    
    with torch.no_grad():
        for data, target in date_loader:
            data, target = data.to(device), target.to(device)
            output = classifier(data)

    return output

def perturb(input, epsilon, grad):

	sign = grad.sign()
	output = input + epsilon * sign
	print(input[0])
	print(output[0])

	return output

def defense(classifier, inversion, device, data_loader):

	classifier.eval()
	inversion.eval()
	epsilon = 0.1

	for batch_idx, (data, target) in enumerate(data_loader):
		data, target = data.to(device), target.to(device)
		with torch.no_grad:
            prediction = classifier(data, release = True)
        
        prediction.requires_grad = True
        reconstruction = inversion(prediction)
        loss =F.mse_loss(reconstruction, target)
        inversion.zere_grad()
        loss.backward()
        prediction_grad = prediction.grad.data
        pert_pred = perturb(prediction, epsilon,prediction_grad)

        return


def inv_test(classifier, inversion, device, data_loader, epoch, msg):

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
    inversion_path = 'out/inv_model_dict.pth'

    try:
        model_checkpoint = torch.load(model_path)
        print(model_checkpoint)
        classifier.load_state_dict(model_checkpoint)

    except:
        print("=> load classifier checkpoint '{}' failed".format(model_path))
        return

    try:
     	inv_checkpoint = torch.load(inversion_path)
        print(inv_checkpoint)
        inversion.load_state_dict(inv_checkpoint)
    except:
    	print("=> load classifier checkpoint '{}' failed".format(inversion_path))
        return


    defense(classifier, inversion, device, celeb_loader)









if __name__ == '__main__':
    main()