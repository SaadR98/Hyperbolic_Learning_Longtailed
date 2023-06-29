#Loading the self-made prototypes


# Define the path to your checkpoint file
checkpoint_path = os.path.join(os.environ['...'], '...')
prototypes = np.load(checkpoint_path)
prototypes = torch.tensor(prototypes)

def separate_prototypes(prototypes):
    nprot = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
    optimizer = optim.SGD([nprot], lr=0.1, momentum=0.9)
    for i in range(1500):
        product = torch.matmul(nprot, nprot.t()) + 1
        product -= 2. * torch.diag(torch.diag(product))
        loss = product.max(dim=1)[0].mean()
        loss.backward()
        optimizer.step()
        nprot = nn.Parameter(F.normalize(nprot, p=2, dim=1))
        optimizer = optim.SGD([nprot], lr=0.1, momentum=0.9)
        cossim = torch.mm(nprot, nprot.t())
        #if i % 250 == 0:
        #    print(i, torch.histogram(cossim, 10))
    #exit()
    return nprot.data

prototypes = separate_prototypes(prototypes)
prototypes = prototypes.cuda() * 0.95
prototypes = prototypes.detach().cpu()


#Identity-matrix prototypes for the Hyperbolic model
#Fill in the number of amount of classes
prototypes = 0.7 * torch.eye(...)

#Identity-matrix prototypes for the Cosine Similarity model
#Fill in the number of amount of classes
prototypes = torch.eye(...)