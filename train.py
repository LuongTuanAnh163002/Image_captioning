import time
from pathlib import Path
import yaml
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import warnings
import platform

from utils.general import increment_path, colorstr, date_modified
from utils.datasets import FlickrDataset, get_data_loader
from models.model import EncoderDecoder

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

warnings.filterwarnings("ignore") #remove warning
def train(opt):
    
    s = f'Seq2seq attention 🚀 {date_modified()} pytorch {torch.__version__} '
    print(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)

    data, epochs, batch_size, learning_rate= opt.data, opt.epochs, opt.batch_size, opt.learning_rate
    num_worker = opt.workers
    embed_size, attention_dim = opt.embed_size, opt.attention_dim
    encoder_dim, decoder_dim = opt.encoder_dim, opt.decoder_dim
    img_size = opt.img_size[0]
    device_number = opt.device

    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    data_dir = data_dict["data_dir"]

    save_dir = opt.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    #----------------------Save hyperparameter--------------------------------
    hyp = {"epchs": epochs, "batch_size": batch_size, 
          "lr": learning_rate, "embed_size": embed_size, "attention_dim":attention_dim,
          "encoder_dim": encoder_dim, "decoder_dim": decoder_dim, "num_worker":num_worker}
    print(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    
    save_yaml = save_dir + "/" + "hyp.yaml"
    with open(save_yaml, 'w') as yaml_file:
        yaml.dump(hyp, yaml_file, default_flow_style=False)
    #----------------------Save hyperparameter--------------------------------


    #----------------------Load dataset--------------------------------
    print(colorstr('Loading dataset from: ') + str(data_dir))
    transforms = T.Compose([T.Resize(img_size),                     
                            T.RandomCrop(224),                 
                            T.ToTensor(),                               
                            T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    
    dataset =  FlickrDataset(root_dir = data_dir+"/Images",
                            captions_file = data_dir+"/captions.txt",
                            transform=transforms)
    with open(save_dir + "/" + 'vocab.pkl', 'wb') as file: #save vocabulary
        pickle.dump(dataset.vocab, file)
    
    data_loader = get_data_loader(dataset=dataset,batch_size=batch_size,
                                    num_workers=num_worker,
                                    shuffle=True,)
    print(colorstr("green", 'Done load dataset'))
    #----------------------Load dataset--------------------------------

    #----------------------Load model--------------------------------
    print("Loading model...")
    vocab_size = len(dataset.vocab)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EncoderDecoder(embed_size=embed_size, vocab_size = vocab_size,
                            attention_dim=attention_dim, encoder_dim=encoder_dim,decoder_dim=decoder_dim, 
                           device=device).to(device)
    
    
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(colorstr("green", "Done load model..."))
    #----------------------Load model--------------------------------

    #----------------------Start training--------------------------------
    print(f"Starting training for {epochs} epochs")
    t0 = time.time()
    train_loss = []
    for epoch in range(1,epochs+1):
        pbar = tqdm(data_loader, leave=True)
        print(('\n' + '%10s' + '%12s' * 2) % ('Epoch', 'gpu_mem', 'train_loss'))
        epoch_loss = 0
        for idx, (image, captions) in enumerate(pbar):
            image,captions = image.to(device),captions.to(device)
            # Zero the gradients.
            optimizer.zero_grad()
            # Feed forward
            outputs,attentions = model(image, captions)
            # Calculate the batch loss.
            targets = captions[:,1:]
            loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' + '%12s' + '%12.4g' * 1) % (
                        '%g/%g' % (epoch, epochs), mem, loss.item())
            pbar.set_description(s)
            epoch_loss += loss.item()
        
        train_loss.append(epoch_loss)
    
    print('%g epochs completed in %.3f hours.\n' % (epochs, (time.time() - t0) / 3600))
    #----------------------End training--------------------------------


    #----------------------Save result--------------------------------
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.plot(train_loss)
    axs.set_title("train_loss")
    
    fig.tight_layout()
    save_path = save_dir + "/" + "results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #----------------------Save result--------------------------------

    #----------------------Save model--------------------------------
    model_state = {
        'num_epochs':epochs,
        'embed_size':embed_size,
        'vocab_size':vocab_size,
        'attention_dim':attention_dim,
        'encoder_dim':encoder_dim,
        'decoder_dim':decoder_dim,
        'state_dict':model.state_dict()
    }
    
    save_weight = save_dir + "/" + "best.pth"
    torch.save(model_state, save_weight)
    #----------------------Save model--------------------------------
    print("All result save in:", save_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, default='data/1class_flag.yaml', help='data.yaml path')
  parser.add_argument('--epochs', type=int, default=300)
  parser.add_argument('--batch_size', type=int, default=32, help='total batch size for all GPUs')
  parser.add_argument('--embed_size', type=int, default=300, help='embed size')
  parser.add_argument('--attention_dim', type=int, default=256, help='attention dim')
  parser.add_argument('--encoder_dim', type=int, default=2048, help='encoder dim')
  parser.add_argument('--decoder_dim', type=int, default=512, help='decoder dim')
  parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning-rate')
  parser.add_argument('--img_size', nargs='+', type=int, default=[226, 226], help='[train, test] image sizes')
  parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
  parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
  parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
  parser.add_argument('--project', default='runs/train', help='save to project/name')
  parser.add_argument('--name', default='exp', help='save to project/name')
  opt = parser.parse_args()
  opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
  print(opt)
  train(opt)