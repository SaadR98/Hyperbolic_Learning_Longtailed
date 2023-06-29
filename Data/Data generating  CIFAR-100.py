#from: https://github.com/dvlab-research/MiSLAS/blob/main/datasets/cifar100.py

# Define root directory / home directory
root = os.environ['....']

class BalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1

        per_cls_weights = 1 / np.array(label_to_count)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        
        
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

class EffectNumSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1

        

        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        
        
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

class RandomCycleIter:

    def __init__ (self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode
        
    def __iter__ (self):
        return self
    
    def __next__ (self):
        self.i += 1
        
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
            
        return self.data_list[self.i]
    
def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):

    i = 0
    j = 0
    while i < n:
        
#         yield next(data_iter_list[next(cls_iter)])
        
        if j >= num_samples_cls:
            j = 0
    
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]]*num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        
        i += 1
        j += 1

class ClassAwareSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, num_samples_cls=4,):
        # pdb.set_trace()
        num_classes = len(np.unique(data_source.targets))
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(data_source.targets):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_cls = num_samples_cls
        
    def __iter__ (self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)
    
    def __len__ (self):
        return self.num_samples
    
def get_sampler():
    return ClassAwareSampler

########## CIFAR 100

class IMBALANCECIFAR100(torchvision.datasets.CIFAR100):
    # class variable to specify the number of classes in the CIFAR100 dataset
    cls_num = 100

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        # call the parent class's __init__ method
        super(IMBALANCECIFAR100, self).__init__(root, train, transform, target_transform, download)
        
        # set the seed for random number generation
        np.random.seed(rand_number)
        # get the number of images per class for the imbalanced dataset
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        # generate the imbalanced dataset
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        # calculate the maximum number of images per class in the original dataset
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        
        # check if the imbalance type is 'exp'
        if imb_type == 'exp':
            # create exponential imbalance
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
                
        # check if the imbalance type is 'step'
        elif imb_type == 'step':
            # create stepwise imbalance
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            # if the imbalance type is not specified or is not 'exp' or 'step',
            # set the number of images per class to the maximum
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        # create new empty lists for the imbalanced dataset's data and targets
        new_data = []
        new_targets = []
        # convert the targets list to a numpy array
        targets_np = np.array(self.targets, dtype=np.int64)
        # get the unique classes in the original dataset
        classes = np.unique(targets_np)
        # create a dictionary to store the number of images per class in the imbalanced dataset
        self.num_per_cls_dict = dict()
        
        # loop through the classes and their corresponding number of images
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            # add the number of images per class to the dictionary
            self.num_per_cls_dict[the_class] = the_img_num
            # get the indices of the current class in the original dataset
            idx = np.where(targets_np == the_class)[0]
            # shuffle the indices
            np.random.shuffle(idx)
            # select the specified number of images for the current class
            selec_idx = idx[:the_img_num]
            # add the selected images to the new data list
            new_data.append(self.data[selec_idx, ...])
            # add the current class as the target for the selected images
            new_targets.extend([the_class, ] * the_img_num)

        # stack the new data list into a numpy array
        new_data = np.vstack(new_data)
        # assign the new data and targets to the class's attributes
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        # create an empty list to store the number of images per class
        cls_num_list = []
        # loop through classes
        for i in range(self.cls_num):
            # add the number of images for each class to the list
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class CIFAR100_LT(object):
    # initializes the class with specified parameters for data loading
    def __init__(self, distributed=False, root= root, imb_type='exp',
                    imb_factor=0.01, batch_size=128, num_works=32):
        # data transform for the training dataset
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # data transform for the evaluation dataset
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # create an imbalanced training dataset
        train_dataset = IMBALANCECIFAR100(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0, train=True, download=True, transform=train_transform)
        # create an evaluation dataset
        eval_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=eval_transform)
        # get the number of images per class in the imbalanced dataset
        self.cls_num_list = train_dataset.get_cls_num_list()
        # create a distributed sampler for the training dataset if specified
        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        # create a data loader for the training dataset
        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, sampler=self.dist_sampler)
        # create a class-aware sampler for the training dataset
        balance_sampler = ClassAwareSampler(train_dataset)
        # create a data loader for the balanced training dataset
        self.train_balance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, sampler=balance_sampler)
        # create a data loader for the evaluation dataset
        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works)


#Define IMB Factor
dataset = CIFAR100_LT(imb_factor=...., batch_size=128) 

train_loader = dataset.train_instance
test_loader  = dataset.eval