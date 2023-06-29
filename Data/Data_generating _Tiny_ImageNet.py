#Downloading Tiny ImageNet and organising the Validation set
#plus the imbalancer for the Tiny ImageNet

url = "..."
target_directory = "..."

command = ["wget", url, "-P", target_directory]
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

if process.returncode == 0:
    print("File downloaded successfully!")
else:
    print("Failed to download the file.")

# Specify the path of the zip file
zip_path = '...'


# Extract the contents of the zip file to the target directory
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(target_directory)

train_dir = '...'
val_dir = '...'
    
trainset = datasets.ImageFolder(train_dir)
testset = datasets.ImageFolder(val_dir)

trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
test = data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Create separate validation subfolders for the validation images based on their labels indicated in the val_annotations txt file
val_img_dir = os.path.join(val_dir, 'images')

# Open and read val annotations text file
fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
data = fp.readlines()

# Create dictionary to store img filename (word 0) and corresponding label (word 1) for every line in the txt file (as key-value pair)
val_img_dict = {}
for line in data:
    words = line.split('\t')
    val_img_dict[words[0]] = words[1]
fp.close()

# Display first 10 entries of resulting val_img_dict dictionary
{k: val_img_dict[k] for k in list(val_img_dict)[:10]}

# Create subfolders (if not present) for validation images based on label, and move images into the respective folders
for img, folder in val_img_dict.items():
    newpath = os.path.join(val_img_dir, folder)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    if os.path.exists(os.path.join(val_img_dir, img)):
        os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))


class IMBALANCETinyImageNet(ImageFolder):
    def __init__(self, root, imb_factor, transform=None, target_transform=None):
        super(IMBALANCETinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)
        self.imb_factor = imb_factor
        self.classes = [d.name for d in os.scandir(root) if d.is_dir()]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_imbalanced_dataset()

    def _make_imbalanced_dataset(self):
        cls_num = len(self.classes)
        img_max = len(self.samples) / cls_num
        img_num_per_cls = []

        if self.imb_factor == 0.0:
            return self.samples

        if self.imb_factor > 0.0:
            # create exponential imbalance
            for cls_idx in range(cls_num):
                num = img_max * (self.imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))

        new_samples = []
        cls_count = {i: 0 for i in range(cls_num)}

        for img_path, target in self.samples:
            cls_idx = target
            if cls_count[cls_idx] < img_num_per_cls[cls_idx]:
                new_samples.append((img_path, target))
                cls_count[cls_idx] += 1

        return new_samples


#https://github.com/DennisHanyuanXu/Tiny-ImageNet/blob/master/src/data_prep.py
train_transform = transforms.Compose(
     [transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# Create the imbalanced Tiny ImageNet dataset
#choose IMB Factor
trainset = IMBALANCETinyImageNet(train_dir, imb_factor=..., transform=train_transform)
testset = ImageFolder(val_dir_dir, transform=test_transform)

#create dataloaders
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)