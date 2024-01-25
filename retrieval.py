import os
import yaml
import clip
import torch
from torch.nn.functional import normalize
from PIL import Image
import time
import shutil
import argparse

def image_text_retrieval(input_images=[], input_text=[], use_embeddings=False, output_base_path='./output', imgLib_dir=None, model_name = None):
    '''
    input_images: paths of images
    input_text: raw text (string)
    use_embeddings: use embeddings to save dataset features or not
    output_base_path: the output path
    imgLib_dir: the image library to search
    '''
    T1 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the model
    try:
        model, preprocess = clip.load(model_name, device)
    except:
        raise Exception(f'Fail to load model, please check model_name. Current model_name: {model_name}')

    preprocessed_images = [preprocess(Image.open(input_image)).unsqueeze(0).to(device) for input_image in input_images]
    input_embeddings_I = [model.encode_image(image) for image in preprocessed_images]

    input_embeddings_T = []
    if len(input_text) > 0:
        tokenized_text = clip.tokenize(input_text).to(device)
        input_embeddings_T = [model.encode_text(tokenized_text)] # torch.shape: [n, 768]

    input_embeddings = input_embeddings_I + input_embeddings_T
    input_embeddings = torch.cat(input_embeddings)
    print(f'input_embeddings shape: {input_embeddings.shape}')

    # retrieval wanted images by embedding
    filedirs = []
    image_embeddings = []
    imgLib_name = os.path.basename(os.path.normpath(imgLib_dir))
    saved_embeddings_path = os.path.join('embeddings', f'{device}-{model_name}-{imgLib_name}_embeddings.pth'.replace('/', '_'))

    if use_embeddings and os.path.isfile(saved_embeddings_path):
        image_embeddings, filedirs = torch.load(saved_embeddings_path)
    else:
        # get image embeddings
        for filename in os.listdir(imgLib_dir):
            f = os.path.join(imgLib_dir, filename)
            image = Image.open(f)
            image_processed = preprocess(image).unsqueeze(0).to(device)
            image_embedding = model.encode_image(image_processed)
            image_embeddings.append(image_embedding.clone().detach())
            filedirs.append(f)

        image_embeddings = torch.cat(image_embeddings)
        if use_embeddings:
            torch.save([image_embeddings, filedirs], saved_embeddings_path)

    # calculate similarity
    sim = image_embeddings @ input_embeddings.T
    sim_norm = normalize(input=sim, p=2, dim=0)
    similarity = sim_norm.sum(dim=1, keepdims=True)
    values, indices = similarity.topk(5, dim=0) # topk similarity

    # save topk images to output directory
    now = time.time()
    output_dir = os.path.join(output_base_path, f'{imgLib_name}-{int(now)}')
    os.makedirs(output_dir, exist_ok=False)
    print('succeed in making output directory!')

    for k, idx in enumerate(indices):
        basename = os.path.basename(filedirs[idx])
        # print('{}\t{}\t{}\t{}'.format(basename, similarity[idx], sim_norm[idx], sim[idx]))
        output_file = os.path.join(output_dir, 'top'+str(k+1)+'-'+'-'+basename)
        shutil.copy(filedirs[idx], output_file)

    T2 = time.time()
    print('running time:\t{} ms'.format(((T2 - T1)) * 1000))
    print('timestamp of output_dir:\t{}'.format(now))

def main(args):
    # load configs from config file
    with open(args.config_file, 'r') as file:
        cfg = yaml.safe_load(file)
        print(f'config from files: {cfg}')
        input_images = cfg['INPUT']['IMAGES']
        input_text = cfg['INPUT']['TEXT']
        output_dir = cfg['OUTPUT_DIR']
        use_embeddings = cfg['DATASET']['USE_EMBEDDINGS']
        imgLib = cfg['DATASET']['IMAGE_LIB']
        model_name = cfg['MODEL']

    if len(input_images) + len(input_text) < 1:
        raise Exception(f'Invalid Input! Input images and text cannot be empty at the same time.')
    
    # retrieval from image dataset
    image_text_retrieval(
        input_images=input_images,
        input_text=input_text,
        use_embeddings=use_embeddings,
        output_base_path=output_dir,
        imgLib_dir=imgLib,
        model_name=model_name,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    args = parser.parse_args()
    print(f'Command Line Args: {args}')

    main(args=args)