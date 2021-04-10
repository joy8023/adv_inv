from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os, shutil
from data import FaceScrub, CelebA, CelebA_out, FaceScrub_out
from model import Classifier, Inversion
import torch.nn.functional as F
import torchvision.utils as vutils

# Training settings
parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch-size', type=int, default=128, metavar='')
parser.add_argument('--test-batch-size', type=int, default=500, metavar='')
parser.add_argument('--epochs', type=int, default=20, metavar='')
parser.add_argument('--lr', type=float, default=0.01, metavar='')
parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--nz', type=int, default=530)
parser.add_argument('--truncation', type=int, default=530)
parser.add_argument('--c', type=float, default=50.)
parser.add_argument('--num_workers', type=int, default=1, metavar='')


def perturb(prediction, epsilon, grad, logit_original):

    sign = grad.sign()
    logit_new = prediction + epsilon * sign 
    
    logit_diff = F.mse_loss(logit_new, logit_original)
    #print('*************logit_diff:',logit_diff.item())

    #calculate accuracy for perturbed images
    original_label = torch.max(logit_original, 1)[1].cpu().numpy()
    new_label = torch.max(logit_new, 1)[1].cpu().numpy()


    orig_label_onehot = F.one_hot(torch.tensor(original_label), 530)
    orig_label_onehot = torch.tensor(orig_label_onehot, dtype=torch.uint8)

    # to keep the max unchanged
    logit_new[orig_label_onehot] = torch.max(logit_new, 1)[0]*1.1

    new_label = torch.max(logit_new, 1)[1].cpu().numpy()

    accu = np.sum(original_label == new_label)/original_label.shape[0]

    output = F.softmax(logit_new, dim=1)
    output_diff = F.mse_loss(output, F.softmax(prediction, dim=1))
    #print('************output_diff', output_diff.item())

    return output

#main outer function
def add_noise(classifier, inversion, device, data_loader, epsilon, num_step):

    classifier.eval()
    inversion.eval()

    for batch_idx, (data_, target_) in enumerate(data_loader):

        data = data_.to(device)
        print('############batch:',batch_idx)


        for i in range(num_step):

            #print('========perturbation iteration:',i)

            with torch.no_grad():
                logit = classifier(data, release = False)
        
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
            perturbation= perturbation.to(device)
            pert_recon = inversion(perturbation)

            #data = pert_recon.clone().detach().to(device)


        #only do the first batch
        #break

    return

def train( inversion, log_interval, device, data_loader, optimizer, epoch):
    #classifier.eval()
    inversion.train()

    for batch_idx, (data, out) in enumerate(data_loader):
        data, out = data.to(device), out.to(device)
        optimizer.zero_grad()
        
        '''
        with torch.no_grad():
            prediction = classifier(data, release=True)
        '''
        
        reconstruction = inversion(F.softmax(out, dim=1))
        loss = F.mse_loss(reconstruction, data)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format( epoch, batch_idx * len(data),
                                                                  len(data_loader.dataset), loss.item()))

        del data, out, loss
        torch.cuda.empty_cache()

def test(inversion, device, data_loader, epoch, msg):
    #classifier.eval()
    inversion.eval()
    mse_loss = 0
    plot = True
    with torch.no_grad():
        for data, out in data_loader:
            data, out = data.to(device), out.to(device)

            #prediction = classifier(data, release=True)
            reconstruction = inversion(F.softmax(out, dim=1))
            mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

            if plot:
                truth = data[0:32]
                inverse = reconstruction[0:32]
                out = torch.cat((inverse, truth))
                for i in range(4):
                    out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
                    out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
                vutils.save_image(out, 'out/recon_{}_def{}.png'.format(msg.replace(" ", ""), epoch), normalize=False)
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
    train_set = CelebA_out('./celeba_def.npz', transform=transform)
    #train_set = CelebA('./celeba_5w_255.nvpy', transform=transform)
    # Inversion attack on TRAIN data of facescrub classifier
    test1_set = FaceScrub_out('./face_def.npz', transform=transform, train=False)
    # Inversion attack on TEST data of facescrub classifier
    test2_set = FaceScrub('./facescrub.npz', transform=transform, train=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test1_loader = torch.utils.data.DataLoader(test1_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test2_loader = torch.utils.data.DataLoader(test2_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    #classifier = nn.DataParallel(Classifier(nc=args.nc, ndf=args.ndf, nz=args.nz)).to(device)
    inversion = nn.DataParallel(Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.truncation, c=args.c)).to(device)
    optimizer = optim.Adam(inversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True)

    # Load classifier
    path = 'model/inv_def.pth'
    
    try:
        checkpoint = torch.load(path)
        #print(checkpoint)
        inversion.load_state_dict(checkpoint)
    except:
        print("=> load classifier checkpoint '{}' failed".format(path))
        return
    

    test(inversion, device, test2_loader, 1, 'test2')
    return

    # Train inversion model
    best_recon_loss = 999
    for epoch in range(1, args.epochs + 1):
        train(inversion, args.log_interval, device, train_loader, optimizer, epoch)
        recon_loss = test( inversion, device, test1_loader, epoch, 'test1')
        #test(classifier, inversion, device, test2_loader, epoch, 'test2')

        if recon_loss < best_recon_loss:
            best_recon_loss = recon_loss
            state = {
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_recon_loss': best_recon_loss
            }
            #torch.save(state, 'model/inversion_def.pth')
            torch.save(inversion.state_dict(), 'model/inv_def.pth')
            shutil.copyfile('out/recon_test1_def{}.png'.format(epoch), 'out/best_test1_def.png')
            #shutil.copyfile('out/recon_test2_{}.png'.format(epoch), 'out/best_test2.png')

if __name__ == '__main__':
    main()
