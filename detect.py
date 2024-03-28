from PIL import Image
import cv2
from pathlib import Path
import pickle
import argparse
import warnings
import platform

from utils.general import increment_path
from models.model import EncoderDecoder
from utils.general import show_image, colorstr, date_modified

import torchvision.transforms as T
import torch

warnings.filterwarnings("ignore") #remove warning
def detect(opt):
  s = f'Seq2seq attention 🚀 {date_modified()} pytorch {torch.__version__} '
  print(colorstr("red", s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s))
  
  source, weight= opt.source, opt.weight
  model_weight = weight + "/" + "best.pth"
  vocab_source = weight + "/" + "vocab.pkl"
  
  embed_size, attention_dim = opt.embed_size, opt.attention_dim
  encoder_dim, decoder_dim = opt.encoder_dim, opt.decoder_dim
  img_size = opt.img_size[0]
  device_number = opt.device

  save_dir = opt.save_dir
  Path(save_dir).mkdir(parents=True, exist_ok=True)
  name_image = Path(source).name
  save_file_img = save_dir + "/" + name_image
  save_file_txt = save_dir + "/" + name_image[:-4] + ".txt"
  
  with open(vocab_source, 'rb') as file:
      vocab = pickle.load(file)
  vocab_size = len(vocab)
  
  transforms = T.Compose([T.Resize(img_size),                     
                          T.RandomCrop(224),                 
                          T.ToTensor(),                               
                          T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
  
  image = Image.open(source).convert("RGB")
  image_copy = cv2.imread(source)
  image = transforms(image)
  image = image.unsqueeze(0)
  img = image[0].detach().clone()
  img1 = image[0].detach().clone()
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint = torch.load(model_weight)
  model = EncoderDecoder(embed_size=embed_size, vocab_size = vocab_size,
                          attention_dim=attention_dim, encoder_dim=encoder_dim,decoder_dim=decoder_dim, 
                          device=device).to(device)
  model.load_state_dict(checkpoint["state_dict"])
  model.eval()
  features_tensors = img.unsqueeze(0)
  with torch.no_grad():
      features = model.encoder(features_tensors.to(device))
      caps,alphas = model.decoder.generate_caption(features,vocab=vocab)
      caption = ' '.join(caps)
      show_image(features_tensors[0],title=caption)
  
  cv2.imwrite(save_file_img, image_copy)
  with open(save_file_txt, "w") as f:
      f.write(caption)
  print("All result save in:", save_dir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', type=str, default='', help='img file, .jpg, .png.....')
  parser.add_argument('--weight', type=str, default='', help='weight .pth of model after training')
  parser.add_argument('--embed_size', type=int, default=300, help='Embed size')
  parser.add_argument('--attention_dim', type=int, default=256, help='Attention dims')
  parser.add_argument('--encoder_dim', type=int, default=2048, help='Encoder dims')
  parser.add_argument('--decoder_dim', type=int, default=512, help='Decoder dims')
  parser.add_argument('--img_size', nargs='+', type=int, default=[226, 226], help='[train, test] image sizes')
  parser.add_argument('--show_image', nargs='?', const=True, default=False, help='show image when run detect')
  parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
  parser.add_argument('--project', default='runs/detect', help='save to project/name')
  parser.add_argument('--name', default='exp', help='save to project/name')
  opt = parser.parse_args()
  opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
  print(colorstr(opt))
  detect(opt)
