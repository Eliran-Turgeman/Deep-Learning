import os
import pathlib
from project.gan import *
import cs236781.plot as plot
import cs236781.download
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import IPython.display
import tqdm
from scipy.interpolate import UnivariateSpline
from IPython.display import Image, display
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



DATA_DIR = pathlib.Path().absolute().joinpath('project/pytorch-datasets')
DEFAULT_DATA_URL = 'http://vis-www.cs.umass.edu/lfw/lfw-a.zip'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMSIZE = 64
STEP = 10

def generateDataSet(data_url=DEFAULT_DATA_URL):
    _, dataset_dir = cs236781.download.download_data(out_path=DATA_DIR, url=data_url, extract=True, force=False)

    tf = T.Compose([
        # Resize to constant spatial dimensions
        T.Resize((IMSIZE, IMSIZE)),
        # PIL.Image -> torch.Tensor
        T.ToTensor(),
        # Dynamic range [0,1] -> [-1, 1]
        T.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
    ])

    ds_gwb = ImageFolder(os.path.dirname(dataset_dir), tf)
    return ds_gwb


def generateResults(train=False, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan_models_by_name = ['Vanila-Base-Gan', 'SN-Gan', 'W-Gan', 'SN-W-Gan']
    results_path = 'project/final-results/'
    
    if train:
        trainModels(gan_models_by_name, num_epochs)
        results_path = 'project/results/'
        
    print('Generating Results')
    for model_name in gan_models_by_name:
            model_results_path = results_path + model_name
            generateResultsFromPath(model_name, model_results_path)
            
    plotScore(results_path, gan_models_by_name, num_epochs)


def plotScore(path, gan_models_by_name, num_epochs):
    x = np.arange(0, num_epochs, STEP)
    xs = np.linspace(0, x[-1] - 1, 100)
    colors = ['k', 'r', 'b', 'g']
    legends = []
    figure(figsize=(14, 8))
    
    for idx, model_name in enumerate(gan_models_by_name):
        scores_file = path + model_name + '/scores.pt'
        
        y = torch.load(scores_file)
        y = [IS[0] for IS in y]
        x = [i*STEP for i in range(0,len(y))]
        xs = np.linspace(0, x[-1] - 1, 100)
        s = UnivariateSpline(x, y, s=5)
        ys = s(xs)
        
        plt.plot(x, y, f'{colors[idx]}.')
        plt.plot(xs, ys, f'{colors[idx]}--')
        legends += [f'{model_name} scores', f'{model_name} trend']
        
    plt.title("Inception Score During Training")
    plt.legend(legends, loc='lower right')
    plt.xlabel("Epoch")
    plt.ylabel("Inception Score ")
    plt.grid(axis='both')
    plt.show()

    
def trainModels(gan_models_by_name, num_epochs=10):
    ds_gwb = generateDataSet()
    x0, y0 = ds_gwb[0]
    x0 = x0.unsqueeze(0).to(device)

    batch_size = DLparams['batch_size']
    im_size = ds_gwb[0][0].shape

    dl_train = DataLoader(ds_gwb, batch_size, shuffle=True)

    ganModels = dict()
    if 'Vanila-Base-Gan' in gan_models_by_name:
        ganModels['Vanila-Base-Gan'] = Gan(im_size, device=device)
    if 'SN-Gan' in gan_models_by_name:
        ganModels['SN-Gan'] = SnGan(im_size, device=device)
    if 'W-Gan' in gan_models_by_name:
        ganModels['W-Gan'] = WGan(im_size, device=device)
    if 'SN-W-Gan' in gan_models_by_name:
        ganModels['SN-W-Gan'] = SnWGan(im_size, device=device)

    try:
        model_file_path = {}
        for gan_name, ganModule in ganModels.items():
            model_file_path[gan_name] = pathlib.Path().parent.absolute().joinpath(
                f'project/results/{gan_name}/')
            if not os.path.exists(model_file_path[gan_name]):
                os.makedirs(model_file_path[gan_name])

        scores, dsc_avg_losses, gen_avg_losses ={}, {}, {}
        for gan_name, ganModule in ganModels.items():
            dsc_avg_losses[gan_name] = []
            gen_avg_losses[gan_name] = []
            scores[gan_name] = []

        for epoch_idx in range(num_epochs):
            # We'll accumulate batch losses and show an average once per epoch.
            dsc_losses, gen_losses = {}, {}
            for gan_name, ganModule in ganModels.items():
                dsc_losses[gan_name] = []
                gen_losses[gan_name] = []
            print(f'--- EPOCH {epoch_idx + 1}/{num_epochs} ---')

            with tqdm.tqdm(total=len(dl_train.batch_sampler), file=sys.stdout) as pbar:
                for batch_idx, (x_data, _) in enumerate(dl_train):
                    x_data = x_data.to(device)
                    for gan_name, ganModule in ganModels.items():
                        dsc_loss, gen_loss = ganModule.trainBatch(x_data)
                        dsc_losses[gan_name].append(dsc_loss)
                        gen_losses[gan_name].append(gen_loss)
                    pbar.update()
            for gan_name, ganModule in ganModels.items():
                gen = ganModule.generator
                dsc_avg_losses[gan_name].append(np.mean(dsc_losses[gan_name]))
                gen_avg_losses[gan_name].append(np.mean(gen_losses[gan_name]))
                if epoch_idx % STEP == 0:
                    print(f'{gan_name}')
                    print(f'Discriminator loss: {dsc_avg_losses[gan_name][-1]}')
                    print(f'Generator loss:     {gen_avg_losses[gan_name][-1]}')

            for gan_name, ganModule in ganModels.items():
                if epoch_idx % STEP == 0:
                    print(f'========{gan_name}========')
                    gen = ganModule.generator
                    samples = gen.sample(5, with_grad=False)
                    fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(6, 2))
                    IPython.display.display(fig)
                    plt.close(fig)
                    print(f'========================')
                    scores[gan_name] += [inception_score(gen, cuda=True, batch_size=32, resize=True, splits=10, len=20000)]
                    torch.save(scores[gan_name], model_file_path[gan_name].joinpath('scores.pt'))
                    print(f'score for{gan_name} is mean: {scores[gan_name][-1][0]} std:{scores[gan_name][-1][1]} ' )


        ganModels = {gan_name: ganModule.generator for gan_name, ganModule in ganModels.items()}
    except KeyboardInterrupt as e:
        print('\n *** Training interrupted by user')

    for gan_name, gen in ganModels.items():
        samples = gen.sample(50, with_grad=False)
        fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(10, 2))
        torch.save(samples, model_file_path[gan_name].joinpath('generated_samples.pt'))
        IS = inception_score(gen, cuda=True, batch_size=32, resize=True, splits=10, len=50000)
        with open(model_file_path[gan_name].joinpath('inception_score.txt'), 'w') as f:
            f.write(str(IS))
        print(f"Inception score for model - is: {IS}")
    print('Training Complete')



def generateResultsFromPath(modelName, modelPath):
    print('=====================================')
    print(f'Model Type: {modelName}')
    with open(f'{modelPath}/inception_score.txt', 'r') as f:
        print(f'Inception Score: {f.read()}')
    print(f'Generated Images: ')
    samples = torch.load(f'{modelPath}/generated_samples.pt')
    fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(15,10), nrows=5)
    IPython.display.display(fig)
    plt.close(fig)
    print('=====================================')