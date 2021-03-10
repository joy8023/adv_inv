from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os, shutil
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
parser.add_argument('--epsilon', type = float, default = 1e-4, metavar = '')
parser.add_argument('--num_step', type = int, default = 10, metavar = '')


def perturb(prediction, epsilon, grad, logit_original):

    sign = grad.sign()
    logit_new = prediction + epsilon * sign 
    
    logit_diff = F.mse_loss(logit_new, logit_original)

    original_label = torch.max(logit_original, 1)[1].cpu().numpy()
    new_label = torch.max(logit_new, 1)[1].cpu().numpy()

    orig_label_onehot = F.one_hot(torch.tensor(original_label), 530)

    orig_label_onehot = torch.tensor(orig_label_onehot, dtype=torch.uint8)

    logit_new[orig_label_onehot] = torch.max(logit_new, 1)[0]*1.1


    new_label = torch.max(logit_new, 1)[1].cpu().numpy()

    accu = np.sum(original_label == new_label)/original_label.shape[0]


    output = F.softmax(logit_new, dim=1)
    output_diff = F.mse_loss(output, F.softmax(prediction, dim=1))
    print('************output_diff', output_diff.item())

    return output

#main outer function
def add_noise(classifier, inversion, device, data_loader, epsilon, num_step):

    classifier.eval()
    inversion.eval()

    for batch_idx, (data_, target_) in enumerate(data_loader):

        data = data_.to(device)
        #print('############batch:',batch_idx)
        '''
        if batch_idx%2 == 1:
            with torch.no_grad():
                pred = classifier(data, release = True)

            img = torch.cat((img,data_))
            result = torch.cat((result,pred))
            print("no defense batch")
            continue
        '''


        for i in range(num_step):

            print('========perturbation iteration:',i)

            with torch.no_grad():
                logit = classifier(data, logit = True)
        
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

            #for plot use
            
            if i == 0:
                truth = data[0:8]
                inverse = reconstruction[0:8]
                out = torch.cat((truth, inverse))
                #out = torch.cat((out, pert_recon[0:8]))

            
            out = torch.cat((out, pert_recon[0:8]))
            
            data = pert_recon.clone().detach().to(device)
        '''
        #to save img and their result
        if batch_idx == 0:
            img = data_
            result = perturbation

        img = torch.cat((img,data_))
        result = torch.cat((result,perturbation))
        print("defense batch")
        '''

        #only do the first batch
        break
    '''
    img = img.cpu().numpy()
    result = result.detach().cpu().numpy()
    print(img.shape)
    print(result.shape)

    #generating dataset
    np.savez("celeba_def_rdm.npz", images = img, out = result)
    '''
    #out = torch.cat((out, torch.tensor(after_noise)))
    vutils.save_image(out, 'out1/test_epsilon_{}.png'.format(epsilon), normalize=False)
    
    return



def train(classifier, inversion, log_interval, device, data_loader, optimizer, epoch):
    classifier.eval()
    inversion.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            prediction = classifier(data)
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
    

def test(classifier, inversion, device, data_loader, epoch, msg):
    classifier.eval()
    inversion.eval()
    mse_loss = 0
    plot = True
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            prediction = classifier(data)
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
    test_set = datasets.MNIST('../data', train=False, download=True,
                       transform=transform)
    #train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    #test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    #test2_loader = torch.utils.data.DataLoader(test2_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    classifier = nn.DataParallel(Net()).to(device)
    inversion = nn.DataParallel(Inversion()).to(device)
    optimizer = optim.Adam(inversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True)

    # Load classifier
    #path = 'out/classifier.pth'
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

    epsilon = args.epsilon
    num_step = args.num_step
    
    add_noise(classifier, inversion, device, test_loader, epsilon, num_step):


if __name__ == '__main__':
    main()
