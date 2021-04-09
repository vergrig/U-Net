device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NiiDataset(Dataset):
    def __init__(self, img_path, tgt_path):
        # load all nii handle in a list
        img_dir = [i for i in os.listdir(img_path) if i[-3:] == "nii"]
        tgt_dir = [i for i in os.listdir(tgt_path) if i[-3:] == "nii"]
        
        self.images_list = []
        self.transforms = transforms.Normalize((0.5,), (0.5,))
        
        for image_path in img_dir:
            tens = self.to_tensor(img_path + '/' + image_path)
            
            for j in range(tens.shape[2]):
                self.images_list.append(tens[:,:,j][None, ...])
            
        self.target_list = []
        
        for image_path in tgt_dir:
            tens = self.to_tensor(tgt_path + '/' + image_path)
            
            for j in range(tens.shape[2]):
                self.target_list.append(tens[:,:,j][None, ...])
                
        print(self.images_list[0].shape,len(self.images_list))
        print(self.target_list[0].shape,len(self.target_list))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        
        classes = torch.cat([self.target_list[idx] == 0, self.target_list[idx] == 1, self.target_list[idx] == 2], 0)
        return self.transforms((self.images_list[idx], classes))
    
    def to_tensor(self, pth):
        return torch.from_numpy(np.asarray(nib.load(pth).dataobj))